"""
Combined Pipeline Endpoint: AlphaFold2 → Docking
This endpoint automates the full pipeline:
1. Run AlphaFold2 prediction on protein sequence
2. Extract rank 1 PDB from results
3. Run docking with SMILES and rank 1 PDB
4. Return combined results

Add this to combined_server.py or use as separate endpoint
"""

@app.route('/pipeline/alphafold2-to-docking', methods=['POST'])
def combined_pipeline():
    """
    Complete pipeline: Protein Sequence → AlphaFold2 → Rank 1 PDB → Docking
    
    Request:
    {
        "protein_sequence": "MKTAYIAKQR...",
        "smiles": "C[C@H](N)C(=O)O"
    }
    
    Response:
    {
        "success": true,
        "alphafold2": {
            "pdb_content": "...",
            "plddt_score": 85.5,
            "jobname": "..."
        },
        "docking": {
            "affinities": [-7.5, -7.2, ...],
            "best_affinity": -7.5,
            "docked_poses": 5,
            "pdbqt_content": "..."
        },
        "pipeline_status": "complete"
    }
    """
    try:
        data = request.get_json() or {}
        protein_sequence = data.get('protein_sequence', '').strip().upper()
        smiles = data.get('smiles', '').strip()
        
        # Validation
        if not protein_sequence:
            return jsonify({'success': False, 'error': 'Protein sequence is required'}), 400
        
        if not smiles:
            return jsonify({'success': False, 'error': 'SMILES string is required'}), 400
        
        valid_amino_acids = set("ACDEFGHIKLMNPQRSTVWY")
        if not all(c in valid_amino_acids for c in protein_sequence):
            return jsonify({'success': False, 'error': 'Invalid amino acid sequence'}), 400
        
        if len(protein_sequence) < 10:
            return jsonify({'success': False, 'error': 'Sequence too short (minimum 10 amino acids)'}), 400
        
        logger.info(f"[Pipeline] Starting combined pipeline:")
        logger.info(f"  - Protein sequence length: {len(protein_sequence)}")
        logger.info(f"  - SMILES length: {len(smiles)}")
        
        # Step 1: Run AlphaFold2 prediction
        logger.info("[Pipeline] Step 1: Running AlphaFold2 prediction...")
        alphafold2_result = None
        try:
            health_status = check_ngrok_health(ALPHAFOLD2_NGROK_URL)
            if health_status['status'] != 'online':
                return jsonify({
                    'success': False,
                    'error': 'AlphaFold2 service is offline. Please ensure the Colab notebook is running.',
                    'step': 'alphafold2_health_check'
                }), 503
            
            ngrok_endpoint = f"{ALPHAFOLD2_NGROK_URL.rstrip('/')}/predict"
            headers = {
                'Content-Type': 'application/json',
                'ngrok-skip-browser-warning': 'true',
                'User-Agent': 'Mozilla/5.0 (compatible; Flask-Backend/1.0)'
            }
            
            response = requests.post(
                ngrok_endpoint,
                json={'sequence': protein_sequence},
                timeout=900,  # 15 minutes for AlphaFold2
                headers=headers,
                allow_redirects=True,
                verify=True
            )
            
            if response.status_code == 200:
                result = response.json()
                if 'error' in result:
                    return jsonify({
                        'success': False,
                        'error': f'AlphaFold2 error: {result["error"]}',
                        'step': 'alphafold2_prediction'
                    }), 500
                
                alphafold2_result = result
                logger.info(f"[Pipeline] ✅ AlphaFold2 completed! pLDDT: {result.get('plddt_score', 'N/A')}")
            else:
                return jsonify({
                    'success': False,
                    'error': f'AlphaFold2 service error: HTTP {response.status_code}',
                    'step': 'alphafold2_prediction'
                }), 500
                
        except requests.exceptions.RequestException as e:
            logger.error(f"[Pipeline] AlphaFold2 connection error: {e}")
            return jsonify({
                'success': False,
                'error': f'Cannot connect to AlphaFold2 service: {str(e)}',
                'step': 'alphafold2_connection'
            }), 503
        
        # Step 2: Extract rank 1 PDB (should already be extracted by AlphaFold2 service)
        pdb_content = alphafold2_result.get('pdb_content', '')
        if not pdb_content:
            return jsonify({
                'success': False,
                'error': 'No PDB content received from AlphaFold2',
                'step': 'pdb_extraction'
            }), 500
        
        logger.info(f"[Pipeline] ✅ Rank 1 PDB extracted ({len(pdb_content)} characters)")
        
        # Step 3: Run Docking with rank 1 PDB
        logger.info("[Pipeline] Step 2: Running molecular docking...")
        docking_result = None
        try:
            health_status = check_ngrok_health(DOCKING_NGROK_URL)
            if health_status['status'] != 'online':
                return jsonify({
                    'success': False,
                    'error': 'Docking service is offline. Please ensure the Colab notebook is running.',
                    'step': 'docking_health_check',
                    'alphafold2': alphafold2_result  # Return AlphaFold2 results even if docking fails
                }), 503
            
            ngrok_endpoint = f"{DOCKING_NGROK_URL.rstrip('/')}/dock"
            payload = {
                'smiles': smiles,
                'protein_pdb': pdb_content  # Use rank 1 PDB from AlphaFold2
            }
            
            response = requests.post(
                ngrok_endpoint,
                json=payload,
                timeout=180,  # 3 minutes for docking
                headers={'Content-Type': 'application/json'}
            )
            
            if response.status_code == 200:
                docking_result = response.json()
                if 'error' in docking_result:
                    return jsonify({
                        'success': False,
                        'error': f'Docking error: {docking_result["error"]}',
                        'step': 'docking_prediction',
                        'alphafold2': alphafold2_result  # Return AlphaFold2 results
                    }), 500
                
                logger.info(f"[Pipeline] ✅ Docking completed! Best affinity: {docking_result.get('best_affinity', 'N/A')} kcal/mol")
            else:
                return jsonify({
                    'success': False,
                    'error': f'Docking service error: HTTP {response.status_code}',
                    'step': 'docking_prediction',
                    'alphafold2': alphafold2_result  # Return AlphaFold2 results
                }), 500
                
        except requests.exceptions.RequestException as e:
            logger.error(f"[Pipeline] Docking connection error: {e}")
            return jsonify({
                'success': False,
                'error': f'Cannot connect to Docking service: {str(e)}',
                'step': 'docking_connection',
                'alphafold2': alphafold2_result  # Return AlphaFold2 results
            }), 503
        
        # Step 4: Return combined results
        logger.info("[Pipeline] ✅ Pipeline completed successfully!")
        return jsonify({
            'success': True,
            'protein_sequence': protein_sequence,
            'smiles': smiles,
            'alphafold2': {
                'pdb_content': alphafold2_result.get('pdb_content', ''),
                'plddt_score': alphafold2_result.get('plddt_score', 0),
                'jobname': alphafold2_result.get('jobname', ''),
                'rank1_file': alphafold2_result.get('rank1_file', 'unknown')
            },
            'docking': docking_result,
            'pipeline_status': 'complete',
            'device_type': device_type,
            'device': str(device)
        })
        
    except Exception as e:
        logger.error(f"[Pipeline] Error: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': f'Pipeline failed: {str(e)}'
        }), 500

