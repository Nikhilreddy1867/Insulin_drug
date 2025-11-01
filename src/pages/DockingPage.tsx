import { useState, useEffect, useRef } from 'react';
import { Loader, CheckCircle, AlertCircle, FlaskConical, Download } from 'lucide-react';

interface DockingResult {
  success: boolean;
  smiles: string;
  result: {
    affinities?: number[];
    best_affinity?: number;
    docked_poses?: number;
    pdbqt_content?: string;
    protein_pdb?: string;
    visualization_url?: string;
    visualization_image?: string;
    visualization_data?: {
      protein_pdb?: string;
      ligand_pdbqt?: string;
    };
    error?: string;
  };
  error?: string;
}

export default function DockingPage() {
  const [smiles, setSmiles] = useState('');
  const [proteinSequence, setProteinSequence] = useState('');
  const [proteinPDB, setProteinPDB] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<DockingResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [useSequence, setUseSequence] = useState(true);

  // 3D Molecule Viewer Component
  function Molecule3DViewer({ proteinPdb, ligandPdbqt }: { proteinPdb: string; ligandPdbqt: string }) {
    const viewerRef = useRef<HTMLDivElement>(null);
    const viewerInstance = useRef<any>(null);

    useEffect(() => {
      if (!viewerRef.current || !proteinPdb || !ligandPdbqt) return;

      // Load 3Dmol.js dynamically
      const load3DMol = async () => {
        try {
          // Use CDN for 3Dmol.js
          if (!(window as any).$3Dmol) {
            const script = document.createElement('script');
            script.src = 'https://cdn.jsdelivr.net/npm/3dmol@2.1.0/build/3Dmol-min.js';
            script.onload = () => initializeViewer();
            document.head.appendChild(script);
          } else {
            initializeViewer();
          }
        } catch (error) {
          console.error('Error loading 3Dmol:', error);
        }
      };

      const initializeViewer = () => {
        if (!viewerRef.current || !(window as any).$3Dmol) return;

        try {
          const viewer = (window as any).$3Dmol.createViewer(viewerRef.current, {
            backgroundColor: 'white',
            defaultZoom: 1.0
          });
          
          // Ensure the canvas doesn't overflow its container
          if (viewerRef.current) {
            const canvas = viewerRef.current.querySelector('canvas');
            if (canvas) {
              canvas.style.position = 'relative';
              canvas.style.maxWidth = '100%';
              canvas.style.maxHeight = '100%';
            }
          }

          // Add protein model (cartoon representation with spectrum coloring)
          if (proteinPdb && proteinPdb.trim()) {
            viewer.addModel(proteinPdb, 'pdb');
            viewer.setStyle({ 'cartoon': { 'color': 'spectrum' } });
          }
          
          // Add ligand model (stick representation)
          // Convert PDBQT to PDB format for 3Dmol.js
          if (ligandPdbqt && ligandPdbqt.trim()) {
            // Extract PDB coordinates from PDBQT (PDBQT format is mostly compatible)
            // Remove REMARK lines and keep ATOM/HETATM lines
            const pdbqtLines = ligandPdbqt.split('\n');
            const pdbLines = pdbqtLines.filter(line => 
              line.startsWith('ATOM') || line.startsWith('HETATM') || line.startsWith('MODEL') || line.startsWith('ENDMDL')
            );
            const ligandPdb = pdbLines.join('\n');
            
            if (ligandPdb.trim()) {
              viewer.addModel(ligandPdb, 'pdb');
              viewer.setStyle({ 'stick': {} });
            }
          }

          viewer.zoomTo();
          viewer.render();
          
          viewerInstance.current = viewer;
        } catch (error) {
          console.error('Error initializing 3D viewer:', error);
        }
      };

      load3DMol();

      return () => {
        if (viewerInstance.current && viewerInstance.current.destroy) {
          viewerInstance.current.destroy();
        }
      };
    }, [proteinPdb, ligandPdbqt]);

    return (
      <div className="w-full relative overflow-hidden rounded-lg" style={{ height: '500px', backgroundColor: 'white', zIndex: 0, isolation: 'isolate' }}>
        <div ref={viewerRef} style={{ width: '100%', height: '100%', position: 'relative', zIndex: 0 }} />
      </div>
    );
  }

  const handleDock = async () => {
    if (!smiles.trim()) {
      setError('Please enter a SMILES string');
      return;
    }
    
    if (useSequence && !proteinSequence.trim()) {
      setError('Please enter a protein sequence');
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const payload: any = {
        smiles: smiles.trim(),
      };

      if (useSequence) {
        payload.protein_sequence = proteinSequence.trim().toUpperCase();
      } else {
        payload.protein_pdb = proteinPDB.trim();
      }

      const response = await fetch('http://localhost:5001/docking/run', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        credentials: 'include',
        body: JSON.stringify(payload),
      });

      if (response.status === 401) {
        setError('Session expired. Please log in again.');
        return;
      }

      const data: DockingResult = await response.json();

      if (data.success) {
        setResult(data);
      } else {
        setError(data.error || 'Docking failed');
      }
    } catch (err) {
      console.error('Docking error:', err);
      setError('Unable to connect to the server. Please ensure the Flask backend and Colab notebook (via ngrok) are running.');
    } finally {
      setLoading(false);
    }
  };

  const downloadPDBQT = () => {
    if (result?.result?.pdbqt_content) {
      const blob = new Blob([result.result.pdbqt_content], { type: 'text/plain' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `docked_poses_${result.smiles.substring(0, 10).replace(/[^a-zA-Z0-9]/g, '_')}.pdbqt`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-900 via-indigo-900 to-purple-900 py-16">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <h1 className="text-4xl font-bold text-white mb-4">Molecular Docking</h1>
        <p className="text-blue-200 mb-12 text-lg">
          Perform molecular docking between drug molecules (SMILES) and protein structures using AutoDock Vina
        </p>

        <div className="bg-white/10 backdrop-blur-sm rounded-xl border border-white/20 p-8 shadow-sm mb-8 space-y-6">
          {/* SMILES Input */}
          <div>
            <label htmlFor="smiles-input" className="block text-xl font-semibold mb-4 text-white">
              SMILES String (Drug Molecule)
            </label>
            <input
              id="smiles-input"
              type="text"
              value={smiles}
              onChange={(e) => setSmiles(e.target.value)}
              placeholder="Enter SMILES string (e.g., C[C@H](N)C(=O)O)"
              className="w-full px-6 py-4 border-2 rounded-xl focus:ring-2 focus:ring-green-500 focus:border-transparent text-base font-mono transition-all duration-300 bg-white/10 border-white/30 text-white placeholder-white/50 focus:ring-green-400"
              disabled={loading}
            />
          </div>

          {/* Protein Input Mode Toggle */}
          <div>
            <label className="flex items-center gap-3 text-white mb-4">
              <input
                type="checkbox"
                checked={useSequence}
                onChange={(e) => setUseSequence(e.target.checked)}
                className="w-4 h-4 rounded"
              />
              <span>Use Protein Sequence (otherwise use PDB file)</span>
            </label>
          </div>

          {/* Protein Sequence or PDB */}
          {useSequence ? (
            <div>
              <label htmlFor="protein-sequence" className="block text-xl font-semibold mb-4 text-white">
                Protein Sequence
              </label>
              <textarea
                id="protein-sequence"
                value={proteinSequence}
                onChange={(e) => setProteinSequence(e.target.value)}
                placeholder="Enter protein sequence or use AlphaFold2 prediction result"
                className="w-full h-32 px-6 py-4 border-2 rounded-xl focus:ring-2 focus:ring-green-500 focus:border-transparent resize-none text-base font-mono leading-relaxed transition-all duration-300 bg-white/10 border-white/30 text-white placeholder-white/50 focus:ring-green-400"
                disabled={loading}
              />
            </div>
          ) : (
            <div>
              <label htmlFor="pdb-upload" className="block text-xl font-semibold mb-4 text-white">
                Protein PDB File Content
              </label>
              <textarea
                id="pdb-upload"
                value={proteinPDB}
                onChange={(e) => setProteinPDB(e.target.value)}
                placeholder="Paste PDB file content here (from AlphaFold2 prediction or upload file)"
                className="w-full h-40 px-6 py-4 border-2 rounded-xl focus:ring-2 focus:ring-green-500 focus:border-transparent resize-none text-base font-mono leading-relaxed transition-all duration-300 bg-white/10 border-white/30 text-white placeholder-white/50 focus:ring-green-400"
                disabled={loading}
              />
              <p className="text-blue-200 text-sm mt-2">
                Tip: Use the PDB file from AlphaFold2 prediction results
              </p>
            </div>
          )}

          {/* Run Docking Button */}
          <div className="flex items-center gap-4">
            <button
              onClick={handleDock}
              disabled={loading || !smiles.trim()}
              className="px-8 py-4 bg-gradient-to-r from-green-500 to-teal-600 hover:from-green-600 hover:to-teal-700 disabled:from-gray-400 disabled:to-gray-500 text-white font-semibold rounded-xl transition-all duration-300 flex items-center justify-center gap-3 shadow-lg hover:shadow-xl transform hover:scale-[1.02] disabled:transform-none disabled:cursor-not-allowed text-lg"
            >
              {loading ? (
                <>
                  <Loader className="w-6 h-6 animate-spin" />
                  Running Docking...
                </>
              ) : (
                <>
                  <FlaskConical className="w-6 h-6" />
                  Run Docking
                </>
              )}
            </button>
            {error && (
              <div className="flex items-center gap-3 p-4 bg-red-50 border border-red-200 rounded-xl animate-fadeIn">
                <AlertCircle className="w-5 h-5 text-red-500 flex-shrink-0" />
                <p className="text-red-700">{error}</p>
              </div>
            )}
          </div>
        </div>

        {result && result.success && (
          <div className="space-y-6 animate-fadeIn">
            <div className="flex items-center gap-3 p-5 bg-green-50 border border-green-200 rounded-xl">
              <CheckCircle className="w-6 h-6 text-green-500 flex-shrink-0" />
              <div>
                <p className="text-green-800 font-semibold text-lg">Docking Completed</p>
                <p className="text-green-600 text-sm">Binding affinity scores calculated</p>
              </div>
            </div>

            <div className="bg-white/10 backdrop-blur-sm rounded-xl border border-white/20 p-8 shadow-sm space-y-6">
              <div className="flex items-center justify-between">
                <h3 className="text-2xl font-bold text-white">Docking Results</h3>
                {result.result.pdbqt_content && (
                  <button
                    onClick={downloadPDBQT}
                    className="px-4 py-2 bg-green-600 hover:bg-green-700 text-white rounded-lg flex items-center gap-2 transition-colors"
                  >
                    <Download className="w-4 h-4" />
                    Download PDBQT
                  </button>
                )}
              </div>

              {result.result.best_affinity !== undefined && (
                <div className="bg-white/5 rounded-lg p-6 border border-white/10">
                  <h4 className="text-lg font-semibold text-white mb-4">Binding Affinity</h4>
                  <div className="text-4xl font-bold text-green-300 mb-2">
                    {result.result.best_affinity.toFixed(2)} kcal/mol
                  </div>
                  <p className="text-blue-200 text-sm">
                    {result.result.best_affinity < -7 ? 'Strong binding' :
                     result.result.best_affinity < -5 ? 'Moderate binding' :
                     'Weak binding'}
                  </p>
                  <p className="text-blue-300 text-xs mt-2">
                    (More negative values indicate stronger binding)
                  </p>
                </div>
              )}

              {result.result.affinities && result.result.affinities.length > 0 && (
                <div className="bg-white/5 rounded-lg p-4 border border-white/10">
                  <h4 className="text-lg font-semibold text-white mb-4">All Pose Affinities</h4>
                  <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
                    {result.result.affinities.map((affinity, idx) => (
                      <div key={idx} className="text-center p-3 bg-white/5 rounded-lg">
                        <div className="text-xs text-blue-200 mb-1">Pose {idx + 1}</div>
                        <div className="text-xl font-bold text-green-300">{affinity.toFixed(2)}</div>
                        <div className="text-xs text-blue-300">kcal/mol</div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {result.result.docked_poses && (
                <div className="bg-white/5 rounded-lg p-4 border border-white/10">
                  <h4 className="text-lg font-semibold text-white mb-2">Generated Poses</h4>
                  <p className="text-blue-200">{result.result.docked_poses} docked conformations generated</p>
                </div>
              )}

              {(result.result.visualization_data || result.result.visualization_url || result.result.visualization_image) && (
                <div className="bg-white/5 rounded-lg p-4 border border-white/10 relative" style={{ zIndex: 0 }}>
                  <h4 className="text-lg font-semibold text-white mb-2">3D Molecular Visualization</h4>
                  <p className="text-blue-200 text-sm mb-3">Interactive 3D view: Click and drag to rotate, scroll to zoom</p>
                  {result.result.visualization_data && (result.result.visualization_data.protein_pdb || result.result.protein_pdb) && result.result.visualization_data.ligand_pdbqt ? (
                    <div className="relative" style={{ zIndex: 0 }}>
                      <Molecule3DViewer
                        proteinPdb={result.result.visualization_data.protein_pdb || result.result.protein_pdb || ''}
                        ligandPdbqt={result.result.visualization_data.ligand_pdbqt || result.result.pdbqt_content || ''}
                      />
                    </div>
                  ) : result.result.visualization_url ? (
                    <iframe
                      src={result.result.visualization_url}
                      className="w-full h-96 rounded-lg border border-white/20"
                      title="Docking Visualization"
                    />
                  ) : result.result.visualization_image ? (
                    <div className="flex justify-center items-center bg-white rounded-lg p-4">
                      <img
                        src={result.result.visualization_image}
                        alt="Ligand Molecular Structure"
                        className="max-w-full h-auto rounded-lg shadow-lg"
                        style={{ maxHeight: '600px' }}
                      />
                    </div>
                  ) : null}
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

