import { useState, useEffect, useRef } from 'react';
import { Send, AlertCircle, CheckCircle, Loader, Activity, LogOut, User, Moon, Sun, Dna, FlaskConical, Download } from 'lucide-react';
import { useNavigate } from 'react-router-dom';

interface User {
  id: string;
  username: string;
}

interface PredictionResult {
  prediction: string;
  confidence: number;
  probabilities: Record<string, number>;
}

interface GeneratedSequence {
  sequence: string;
  average_probability: number;
  levenshtein: number;
  hamming: number;
}

interface ApiResponse {
  success: boolean;
  result: PredictionResult;
  processed_sequence: string;
  device_type?: string;
  device?: string;
  error?: string;
}

interface SequenceApiResponse {
  success: boolean;
  input_sequence: string;
  total_generated?: number;
  top_sequences: GeneratedSequence[];
  metrics_used?: string[];
  ranking?: string;
  device_type?: string;
  device?: string;
  error?: string;
}

interface SmilesApiResponse {
  success: boolean;
  input_sequence: string;
  smiles: string;
  device_type?: string;
  device?: string;
  deviceType?: string;
  error?: string;
}

export default function DashboardPage({ user, onLogout, isDarkMode, toggleTheme }: {
  user: User | null;
  onLogout: () => void;
  isDarkMode: boolean;
  toggleTheme: () => void;
}) {
  const navigate = useNavigate();

  if (!user) {
    navigate('/');
    return null;
  }
  const [sequence, setSequence] = useState('');
  const [predictionLoading, setPredictionLoading] = useState(false);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [predictionDevice, setPredictionDevice] = useState<string>('');
  const [generatorSequence, setGeneratorSequence] = useState('');
  const [sequenceGenerationLoading, setSequenceGenerationLoading] = useState(false);
  const [generatedSequences, setGeneratedSequences] = useState<GeneratedSequence[]>([]);
  const [sequenceGenerationError, setSequenceGenerationError] = useState<string | null>(null);
  const [sequenceDevice, setSequenceDevice] = useState<string>('');
  const [smilesSequence, setSmilesSequence] = useState('');
  const [smilesLoading, setSmilesLoading] = useState(false);
  const [generatedSmiles, setGeneratedSmiles] = useState<string>('');
  const [smilesError, setSmilesError] = useState<string | null>(null);
  const [smilesDevice, setSmilesDevice] = useState<string>('');
  const [activeTab, setActiveTab] = useState<'classification' | 'sequence-generation' | 'smiles-generation' | 'alphafold2' | 'docking' | 'pipeline'>('classification');
  
  // AlphaFold2 state
  const [alphafoldSequence, setAlphafoldSequence] = useState('');
  const [alphafoldLoading, setAlphafoldLoading] = useState(false);
  const [alphafoldResult, setAlphafoldResult] = useState<any>(null);
  const [alphafoldError, setAlphafoldError] = useState<string | null>(null);
  
  // Docking state
  const [dockingSmiles, setDockingSmiles] = useState('');
  const [dockingProteinPDB, setDockingProteinPDB] = useState('');
  const [dockingLoading, setDockingLoading] = useState(false);
  const [dockingResult, setDockingResult] = useState<any>(null);
  const [dockingError, setDockingError] = useState<string | null>(null);
  const [useProteinSequence, setUseProteinSequence] = useState(false);
  const [dockingProteinSequence, setDockingProteinSequence] = useState('');
  
  // Pipeline state
  const [pipelineSequence, setPipelineSequence] = useState('');
  const [pipelineSmiles, setPipelineSmiles] = useState('');
  const [pipelineLoading, setPipelineLoading] = useState(false);
  const [pipelineResult, setPipelineResult] = useState<any>(null);
  const [pipelineError, setPipelineError] = useState<string | null>(null);
  const [pipelineStep, setPipelineStep] = useState<string>('');

  const validateSequence = (seq: string): boolean => {
    const validAminoAcids = /^[ACDEFGHIKLMNPQRSTVWY]+$/i;
    return validAminoAcids.test(seq.trim());
  };

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

  const handlePredict = async () => {
    const trimmedSequence = sequence.trim().toUpperCase();
    
    if (!trimmedSequence) {
      setError('Please enter a protein sequence');
      return;
    }
    
    if (trimmedSequence.length < 10) {
      setError('Sequence too short. Please provide at least 10 amino acids.');
      return;
    }
    
    if (!validateSequence(trimmedSequence)) {
      setError('Invalid amino acid sequence. Please use single-letter amino acid codes (A-Z, excluding B, J, O, U, X, Z).');
      return;
    }

    setPredictionLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await fetch('http://localhost:5001/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        credentials: 'include',
        body: JSON.stringify({ sequence: trimmedSequence }),
      });

      if (response.status === 401) {
        setError('Session expired. Please log in again.');
        return;
      }

      const data: ApiResponse = await response.json();

      if (data.success && data.result) {
        setResult(data.result);
        setPredictionDevice(data.device_type || data.device || 'Unknown');
      } else {
        setError(data.error || 'Prediction failed');
      }
    } catch (err) {
      console.error('Prediction error:', err);
      setError('Unable to connect to the server. Please ensure the Flask backend is running.');
    } finally {
      setPredictionLoading(false);
    }
  };

  const handleGenerateSequences = async () => {
    const trimmedSequence = generatorSequence.trim().toUpperCase();
    
    if (!trimmedSequence) {
      setSequenceGenerationError('Please enter a protein sequence');
      return;
    }
    
    if (trimmedSequence.length < 10) {
      setSequenceGenerationError('Sequence too short. Please provide at least 10 amino acids.');
      return;
    }
    
    if (!validateSequence(trimmedSequence)) {
      setSequenceGenerationError('Invalid amino acid sequence. Please use single-letter amino acid codes (A-Z, excluding B, J, O, U, X, Z).');
      return;
    }

    setSequenceGenerationLoading(true);
    setSequenceGenerationError(null);
    setGeneratedSequences([]);

    try {
      const response = await fetch('http://localhost:5001/generate-sequences', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        credentials: 'include',
        body: JSON.stringify({ sequence: trimmedSequence }),
      });

      if (response.status === 401) {
        setSequenceGenerationError('Session expired. Please log in again.');
        return;
      }

      const data: SequenceApiResponse = await response.json();

      if (data.success && data.top_sequences) {
        setGeneratedSequences(data.top_sequences);
        setSequenceDevice(data.device_type || data.device || 'Unknown');
      } else {
        setSequenceGenerationError(data.error || 'Sequence generation failed');
      }
    } catch (err) {
      console.error('Sequence generation error:', err);
      setSequenceGenerationError('Unable to connect to the server. Please ensure the Flask backend is running.');
    } finally {
      setSequenceGenerationLoading(false);
    }
  };

  const handleGenerateSmiles = async () => {
    const trimmedSequence = smilesSequence.trim().toUpperCase();
    
    if (!trimmedSequence) {
      setSmilesError('Please enter a protein sequence');
      return;
    }
    
    if (trimmedSequence.length < 10) {
      setSmilesError('Sequence too short. Please provide at least 10 amino acids.');
      return;
    }
    
    if (!validateSequence(trimmedSequence)) {
      setSmilesError('Invalid amino acid sequence. Please use single-letter amino acid codes (A-Z, excluding B, J, O, U, X, Z).');
      return;
    }

    setSmilesLoading(true);
    setSmilesError(null);
    setGeneratedSmiles('');

    try {
      const response = await fetch('http://localhost:5001/generate-smiles', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        credentials: 'include',
        body: JSON.stringify({ sequence: trimmedSequence }),
      });

      if (response.status === 401) {
        setSmilesError('Session expired. Please log in again.');
        return;
      }

      const data: SmilesApiResponse = await response.json();

      if (data.success && data.smiles) {
        setGeneratedSmiles(data.smiles);
        setSmilesDevice(data.deviceType || data.device_type || data.device || 'Unknown');
      } else {
        setSmilesError(data.error || 'SMILES generation failed');
      }
    } catch (err) {
      console.error('SMILES generation error:', err);
      setSmilesError('Unable to connect to the server. Please ensure the Flask backend is running.');
    } finally {
      setSmilesLoading(false);
    }
  };

  const getSeverityColor = (prediction: string): string => {
    const severityColors: Record<string, string> = {
      'Pathogenic': 'text-red-500',
      'Likely_pathogenic': 'text-orange-500',
      'Uncertain_significance': 'text-yellow-500',
      'Likely_benign': 'text-blue-500',
      'Benign': 'text-green-500',
    };
    return severityColors[prediction] || 'text-gray-500';
  };

  const formatPrediction = (prediction: string): string => {
    return prediction.replace(/_/g, ' ');
  };

  // Pipeline handler
  const handlePipelineRun = async () => {
    const trimmedSequence = pipelineSequence.trim().toUpperCase();
    const trimmedSmiles = pipelineSmiles.trim();
    
    if (!trimmedSequence) {
      setPipelineError('Please enter a protein sequence');
      return;
    }
    
    if (!trimmedSmiles) {
      setPipelineError('Please enter a SMILES string');
      return;
    }
    
    if (trimmedSequence.length < 10) {
      setPipelineError('Sequence too short. Please provide at least 10 amino acids.');
      return;
    }
    
    if (!validateSequence(trimmedSequence)) {
      setPipelineError('Invalid amino acid sequence. Please use single-letter amino acid codes.');
      return;
    }

    setPipelineLoading(true);
    setPipelineError(null);
    setPipelineResult(null);
    setPipelineStep('Starting pipeline...');

    try {
      // Create AbortController for timeout handling (pipeline can take 15-20 minutes)
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 1200000); // 20 minutes timeout

      const response = await fetch('http://localhost:5001/pipeline/alphafold2-to-docking', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        credentials: 'include',
        body: JSON.stringify({
          protein_sequence: trimmedSequence,
          smiles: trimmedSmiles
        }),
        signal: controller.signal
      });

      clearTimeout(timeoutId);

      if (response.status === 401) {
        setPipelineError('Session expired. Please log in again.');
        return;
      }

      const data = await response.json();

      if (data.success) {
        setPipelineResult(data);
        setPipelineStep('Pipeline completed successfully!');
      } else {
        setPipelineError(data.error || 'Pipeline failed');
        if (data.alphafold2) {
          // Partial success - AlphaFold2 completed but docking failed
          setPipelineResult(data);
        }
      }
    } catch (err: any) {
      console.error('Pipeline error:', err);
      if (err.name === 'AbortError' || err.message?.includes('timeout')) {
        setPipelineError('Request timeout. Pipeline can take 15-20 minutes. Please try again with a shorter sequence for testing.');
      } else {
        setPipelineError('Unable to connect to the server. Please ensure both Colab notebooks (AlphaFold2 and Docking) are running.');
      }
    } finally {
      setPipelineLoading(false);
      setPipelineStep('');
    }
  };

  // AlphaFold2 handler
  const handleAlphaFold2Predict = async () => {
    const trimmedSequence = alphafoldSequence.trim().toUpperCase();
    
    if (!trimmedSequence) {
      setAlphafoldError('Please enter a protein sequence');
      return;
    }
    
    if (trimmedSequence.length < 10) {
      setAlphafoldError('Sequence too short. Please provide at least 10 amino acids.');
      return;
    }
    
    if (!validateSequence(trimmedSequence)) {
      setAlphafoldError('Invalid amino acid sequence. Please use single-letter amino acid codes.');
      return;
    }

    setAlphafoldLoading(true);
    setAlphafoldError(null);
    setAlphafoldResult(null);

    try {
      // Create AbortController for timeout handling
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 900000); // 15 minutes timeout
      
      const response = await fetch('http://localhost:5001/alphafold2/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        credentials: 'include',
        body: JSON.stringify({ sequence: trimmedSequence }),
        signal: controller.signal,
      });
      
      clearTimeout(timeoutId);

      if (response.status === 401) {
        setAlphafoldError('Session expired. Please log in again.');
        return;
      }

      const data = await response.json();

      if (data.success) {
        setAlphafoldResult(data);
      } else {
        setAlphafoldError(data.error || 'AlphaFold2 prediction failed');
      }
    } catch (err: any) {
      console.error('AlphaFold2 error:', err);
      if (err.name === 'AbortError' || err.message?.includes('timeout')) {
        setAlphafoldError('Request timeout. AlphaFold2 predictions can take 5-15 minutes. Please try again with a shorter sequence for testing.');
      } else {
        setAlphafoldError('Unable to connect to the server. Please ensure the Flask backend and Colab notebook (via ngrok) are running.');
      }
    } finally {
      setAlphafoldLoading(false);
    }
  };

  const downloadAlphaFoldPDB = () => {
    if (alphafoldResult?.result?.pdb_content) {
      const blob = new Blob([alphafoldResult.result.pdb_content], { type: 'text/plain' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `alphafold2_prediction_${alphafoldResult.sequence.substring(0, 10)}.pdb`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    }
  };

  // Docking handler
  const handleDocking = async () => {
    if (!dockingSmiles.trim()) {
      setDockingError('Please enter a SMILES string');
      return;
    }
    
    if (!useProteinSequence && !dockingProteinPDB.trim()) {
      setDockingError('Please provide protein PDB content (from AlphaFold2) or sequence');
      return;
    }

    if (useProteinSequence && !dockingProteinSequence.trim()) {
      setDockingError('Please enter a protein sequence');
      return;
    }

    setDockingLoading(true);
    setDockingError(null);
    setDockingResult(null);

    try {
      const payload: any = {
        smiles: dockingSmiles.trim(),
      };

      if (useProteinSequence) {
        payload.protein_sequence = dockingProteinSequence.trim().toUpperCase();
      } else {
        payload.protein_pdb = dockingProteinPDB.trim();
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
        setDockingError('Session expired. Please log in again.');
        return;
      }

      const data = await response.json();

      if (data.success) {
        setDockingResult(data);
      } else {
        setDockingError(data.error || 'Docking failed');
      }
    } catch (err) {
      console.error('Docking error:', err);
      setDockingError('Unable to connect to the server. Please ensure the Flask backend and Colab notebook (via ngrok) are running.');
    } finally {
      setDockingLoading(false);
    }
  };

  const downloadDockingPDBQT = () => {
    if (dockingResult?.result?.pdbqt_content) {
      const blob = new Blob([dockingResult.result.pdbqt_content], { type: 'text/plain' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `docked_poses_${dockingResult.smiles.substring(0, 10).replace(/[^a-zA-Z0-9]/g, '_')}.pdbqt`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    }
  };

  return (
    <div className={`min-h-screen relative overflow-hidden transition-all duration-500 ease-in-out ${isDarkMode ? 'bg-gradient-to-br from-slate-900 via-blue-950 to-indigo-950' : 'bg-gradient-to-br from-blue-900 via-indigo-900 to-purple-900'}`}>
      {/* Background particles effect */}
      <div className="absolute inset-0 overflow-hidden">
        <div className={`absolute top-20 left-20 w-2 h-2 rounded-full opacity-20 animate-pulse transition-colors duration-500 ${isDarkMode ? 'bg-blue-400' : 'bg-white'}`}></div>
        <div className={`absolute top-40 right-32 w-1 h-1 rounded-full opacity-30 animate-pulse transition-colors duration-500 ${isDarkMode ? 'bg-purple-400' : 'bg-blue-300'}`}></div>
        <div className={`absolute bottom-32 left-16 w-2 h-2 rounded-full opacity-25 animate-pulse transition-colors duration-500 ${isDarkMode ? 'bg-teal-400' : 'bg-purple-300'}`}></div>
      </div>

      <div className="relative z-10 min-h-screen flex flex-col items-center justify-center p-4">
        {/* Header with user info, theme toggle, and logout */}
        <div className="absolute top-4 right-4 flex items-center gap-4">
          <div className={`flex items-center gap-2 backdrop-blur-sm rounded-full px-4 py-2 border transition-all duration-300 ${isDarkMode ? 'bg-slate-800/60 border-blue-600 shadow-lg' : 'bg-white/10 border-white/20'}`}>
            <User className={`w-4 h-4 transition-colors duration-300 ${isDarkMode ? 'text-blue-300' : 'text-blue-300'}`} />
            <span className={`text-sm font-medium transition-colors duration-300 ${isDarkMode ? 'text-blue-100' : 'text-white'}`}>{user.username}</span>
          </div>
          <button
            onClick={toggleTheme}
            className={`rounded-full p-2 transition-all duration-300 hover:scale-110 ${isDarkMode ? 'bg-indigo-700 hover:bg-indigo-600 text-indigo-200 shadow-lg' : 'bg-white/10 hover:bg-white/20 text-blue-300'}`}
            title={isDarkMode ? 'Switch to Light Mode' : 'Switch to Dark Mode'}
          >
            {isDarkMode ? <Sun className="w-4 h-4" /> : <Moon className="w-4 h-4" />}
          </button>
          <button
            onClick={onLogout}
            className="bg-red-500 hover:bg-red-600 text-white rounded-full p-2 transition-all duration-300 hover:scale-110 shadow-lg"
            title="Logout"
          >
            <LogOut className="w-4 h-4" />
          </button>
        </div>

        {/* Header */}
        <div className="text-center mb-8">
          <div className="flex items-center justify-center mb-6">
            <div className={`backdrop-blur-sm rounded-full p-4 mr-4 border transition-all duration-300 ${isDarkMode ? 'bg-indigo-800/60 border-indigo-600 shadow-lg' : 'bg-white/10 border-white/20'}`}>
              <Activity className={`w-10 h-10 transition-colors duration-300 ${isDarkMode ? 'text-indigo-300' : 'text-blue-300'}`} />
            </div>
            <h1 className={`text-5xl md:text-6xl font-bold tracking-tight transition-colors duration-300 ${isDarkMode ? 'text-blue-100' : 'text-white'}`}>
              Insulin T2D Drug Analysis
            </h1>
          </div>
          <p className={`text-xl md:text-2xl font-light max-w-3xl mx-auto transition-colors duration-300 ${isDarkMode ? 'text-blue-200' : 'text-blue-200'}`}>
            Advanced AI-powered protein sequence pathogenicity classifier
          </p>
        </div>

        {/* Navigation Bar */}
        <div className="w-full max-w-7xl mb-8">
          <div className="flex flex-wrap justify-center gap-4">
            <button
              onClick={() => setActiveTab('classification')}
              className={`px-8 py-4 rounded-xl font-semibold text-lg transition-all duration-300 transform hover:scale-105 ${
                activeTab === 'classification'
                  ? 'bg-gradient-to-r from-blue-500 to-indigo-600 text-white shadow-lg'
                  : `${isDarkMode ? 'bg-slate-800/60 border border-slate-600 text-slate-200 hover:bg-slate-700/60' : 'bg-white/10 border border-white/20 text-white hover:bg-white/20'}`
              }`}
            >
              Classification
            </button>
            <button
              onClick={() => setActiveTab('sequence-generation')}
              className={`px-8 py-4 rounded-xl font-semibold text-lg transition-all duration-300 transform hover:scale-105 ${
                activeTab === 'sequence-generation'
                  ? 'bg-gradient-to-r from-purple-500 to-pink-600 text-white shadow-lg'
                  : `${isDarkMode ? 'bg-slate-800/60 border border-slate-600 text-slate-200 hover:bg-slate-700/60' : 'bg-white/10 border border-white/20 text-white hover:bg-white/20'}`
              }`}
            >
              Sequence Generation
            </button>
            <button
              onClick={() => setActiveTab('smiles-generation')}
              className={`px-8 py-4 rounded-xl font-semibold text-lg transition-all duration-300 transform hover:scale-105 ${
                activeTab === 'smiles-generation'
                  ? 'bg-gradient-to-r from-green-500 to-teal-600 text-white shadow-lg'
                  : `${isDarkMode ? 'bg-slate-800/60 border border-slate-600 text-slate-200 hover:bg-slate-700/60' : 'bg-white/10 border border-white/20 text-white hover:bg-white/20'}`
              }`}
            >
              SMILES Generation
            </button>
            <button
              onClick={() => setActiveTab('alphafold2')}
              className={`px-8 py-4 rounded-xl font-semibold text-lg transition-all duration-300 transform hover:scale-105 ${
                activeTab === 'alphafold2'
                  ? 'bg-gradient-to-r from-indigo-500 to-blue-600 text-white shadow-lg'
                  : `${isDarkMode ? 'bg-slate-800/60 border border-slate-600 text-slate-200 hover:bg-slate-700/60' : 'bg-white/10 border border-white/20 text-white hover:bg-white/20'}`
              }`}
            >
              AlphaFold2
            </button>
            <button
              onClick={() => setActiveTab('docking')}
              className={`px-8 py-4 rounded-xl font-semibold text-lg transition-all duration-300 transform hover:scale-105 ${
                activeTab === 'docking'
                  ? 'bg-gradient-to-r from-teal-500 to-cyan-600 text-white shadow-lg'
                  : `${isDarkMode ? 'bg-slate-800/60 border border-slate-600 text-slate-200 hover:bg-slate-700/60' : 'bg-white/10 border border-white/20 text-white hover:bg-white/20'}`
              }`}
            >
              Docking
            </button>
            <button
              onClick={() => setActiveTab('pipeline')}
              className={`px-8 py-4 rounded-xl font-semibold text-lg transition-all duration-300 transform hover:scale-105 ${
                activeTab === 'pipeline'
                  ? 'bg-gradient-to-r from-orange-500 to-red-600 text-white shadow-lg'
                  : `${isDarkMode ? 'bg-slate-800/60 border border-slate-600 text-slate-200 hover:bg-slate-700/60' : 'bg-white/10 border border-white/20 text-white hover:bg-white/20'}`
              }`}
            >
              Complete Pipeline
            </button>
          </div>
        </div>

        {/* Main Card - Content continues from original App.tsx */}
        <div className="w-full max-w-7xl">
          {/* Classification Tab Content */}
          {activeTab === 'classification' && (
            <div className={`p-8 rounded-xl mb-12 transition-all duration-300 ${isDarkMode ? 'bg-slate-800/90 border border-indigo-600' : 'bg-white/95 border border-white/10'}`}>
              <label htmlFor="sequence" className={`block text-xl font-semibold mb-4 transition-colors duration-300 ${isDarkMode ? 'text-blue-100' : 'text-gray-800'}`}>Protein Sequence</label>
              <textarea
                id="sequence"
                value={sequence}
                onChange={(e) => setSequence(e.target.value)}
                placeholder="Enter your protein sequence here..."
                className={`w-full h-36 px-6 py-4 border-2 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none text-base font-mono leading-relaxed transition-all duration-300 hover:border-gray-300 ${isDarkMode ? 'bg-slate-700 border-indigo-600 text-blue-100 placeholder-blue-300 focus:ring-blue-400' : 'border-gray-200 text-gray-800 placeholder-gray-600'}`}
                disabled={predictionLoading}
              />
              <div className="mt-4 flex flex-col md:flex-row md:items-center md:gap-6 gap-3">
                <button
                  onClick={handlePredict}
                  disabled={predictionLoading || !sequence.trim()}
                  className="w-full md:w-max bg-gradient-to-r from-blue-500 to-indigo-600 hover:from-blue-600 hover:to-indigo-700 disabled:from-gray-400 disabled:to-gray-500 text-white font-semibold py-4 px-8 rounded-xl transition-all duration-300 flex items-center justify-center gap-3 shadow-lg hover:shadow-xl transform hover:scale-[1.02] disabled:transform-none disabled:cursor-not-allowed text-lg"
                >
                  {predictionLoading ? (<><Loader className="w-6 h-6 animate-spin" />Analyzing...</>) : (<><Send className="w-6 h-6" />Predict</>)}
                </button>
                {error && <div className="flex items-center gap-3 p-4 bg-red-50 border border-red-200 rounded-xl animate-fadeIn"><AlertCircle className="w-5 h-5 text-red-500 flex-shrink-0" /><p className="text-red-700">{error}</p></div>}
              </div>
              {result && (
                <div className="space-y-6 animate-fadeIn mt-6">
                  <div className="flex items-center gap-3 p-5 bg-green-50 border border-green-200 rounded-xl">
                    <CheckCircle className="w-6 h-6 text-green-500 flex-shrink-0" />
                    <div className="flex-1">
                      <p className="text-green-800 font-semibold text-lg">Prediction Complete</p>
                      <p className="text-green-600 text-sm">Analysis finished successfully</p>
                    </div>
                    {predictionDevice && (
                      <div className={`px-3 py-1 rounded-lg text-xs font-semibold ${
                        predictionDevice.includes('GPU') ? 'bg-blue-100 text-blue-800' : 'bg-gray-100 text-gray-800'
                      }`}>
                        {predictionDevice}
                      </div>
                    )}
                  </div>

                  <div className={`rounded-xl p-8 space-y-6 transition-all duration-300 ${isDarkMode ? 'bg-slate-700/50 border border-indigo-600' : 'bg-gray-50'}`}>
                    <div className="grid md:grid-cols-2 gap-8">
                      <div className="space-y-3">
                        <h3 className={`text-xl font-bold transition-colors duration-300 ${isDarkMode ? 'text-blue-100' : 'text-gray-800'}`}>Prediction</h3>
                        <p className={`text-3xl font-bold ${getSeverityColor(result.prediction)}`}>
                          {formatPrediction(result.prediction)}
                        </p>
                        <p className={`text-lg transition-colors duration-300 ${isDarkMode ? 'text-blue-200' : 'text-gray-600'}`}>
                          Confidence: <span className="font-semibold">{(result.confidence * 100).toFixed(1)}%</span>
                        </p>
                      </div>

                      <div className="space-y-3">
                        <h3 className={`text-xl font-bold transition-colors duration-300 ${isDarkMode ? 'text-blue-100' : 'text-gray-800'}`}>All Probabilities</h3>
                        <div className="space-y-3">
                          {Object.entries(result.probabilities)
                            .sort(([,a], [,b]) => b - a)
                            .map(([className, probability]) => (
                              <div key={className} className="flex justify-between items-center py-1">
                                <span className={`font-medium transition-colors duration-300 ${isDarkMode ? 'text-blue-200' : 'text-gray-700'}`}>
                                  {formatPrediction(className)}
                                </span>
                                <span className={`font-semibold transition-colors duration-300 ${isDarkMode ? 'text-blue-100' : 'text-gray-800'}`}>
                                  {(probability * 100).toFixed(1)}%
                                </span>
                              </div>
                            ))}
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Sequence Generation Tab Content */}
          {activeTab === 'sequence-generation' && (
          <div className={`p-8 rounded-xl mb-12 transition-all duration-300 ${isDarkMode ? 'bg-purple-950/80 border border-purple-600' : 'bg-purple-50/90 border border-purple-100'}`}>
            <label htmlFor="generatorSequence" className={`block text-xl font-semibold mb-4 transition-colors duration-300 ${isDarkMode ? 'text-purple-100' : 'text-purple-800'}`}>Input Protein Sequence for Generation</label>
            <textarea
              id="generatorSequence"
              value={generatorSequence}
              onChange={(e) => setGeneratorSequence(e.target.value)}
              placeholder="Enter protein sequence to generate new variants..."
              className={`w-full h-32 px-6 py-4 border-2 rounded-xl focus:ring-2 focus:ring-purple-500 focus:border-transparent resize-none text-base font-mono leading-relaxed transition-all duration-300 hover:border-gray-300 ${isDarkMode ? 'bg-slate-700 border-purple-600 text-purple-100 placeholder-purple-300 focus:ring-purple-400' : 'border-gray-200 text-gray-800 placeholder-gray-600'}`}
              disabled={sequenceGenerationLoading}
            />
            <div className="mt-4 flex flex-col md:flex-row md:items-center md:gap-6 gap-3">
              <button
                onClick={handleGenerateSequences}
                disabled={sequenceGenerationLoading || !generatorSequence.trim()}
                className="w-full md:w-max bg-gradient-to-r from-purple-500 to-pink-600 hover:from-purple-600 hover:to-pink-700 disabled:from-gray-400 disabled:to-gray-500 text-white font-semibold py-4 px-8 rounded-xl transition-all duration-300 flex items-center justify-center gap-3 shadow-lg hover:shadow-xl transform hover:scale-[1.02] disabled:transform-none disabled:cursor-not-allowed text-lg"
              >
                {sequenceGenerationLoading ? (<><Loader className="w-6 h-6 animate-spin" />Generating...</>) : (<><Activity className="w-6 h-6" />Generate Sequences</>)}
              </button>
              {sequenceGenerationError && <div className="flex items-center gap-3 p-4 bg-red-50 border border-red-200 rounded-xl animate-fadeIn"><AlertCircle className="w-5 h-5 text-red-500 flex-shrink-0" /><p className="text-red-700">{sequenceGenerationError}</p></div>}
            </div>
            {generatedSequences.length > 0 && (
                <div className="space-y-6 animate-fadeIn mt-6">
                  <div className="flex items-center gap-3 p-5 bg-purple-50 border border-purple-200 rounded-xl">
                    <CheckCircle className="w-6 h-6 text-purple-500 flex-shrink-0" />
                    <div className="flex-1">
                      <p className="text-purple-800 font-semibold text-lg">Sequence Generation Complete</p>
                      <p className="text-purple-600 text-sm">Generated {generatedSequences.length} sequences with similarity analysis</p>
                    </div>
                    {sequenceDevice && (
                      <div className={`px-3 py-1 rounded-lg text-xs font-semibold ${
                        sequenceDevice.includes('GPU') ? 'bg-blue-100 text-blue-800' : 'bg-gray-100 text-gray-800'
                      }`}>
                        {sequenceDevice}
                      </div>
                    )}
                  </div>

                  <div className={`rounded-xl p-8 space-y-6 transition-all duration-300 ${isDarkMode ? 'bg-slate-700/50 border border-purple-600' : 'bg-purple-50'}`}>
                    <div className="text-center">
                      <h3 className={`text-2xl font-bold mb-2 transition-colors duration-300 ${isDarkMode ? 'text-purple-100' : 'text-purple-800'}`}>
                        Generated Protein Sequences
                      </h3>
                      <p className={`text-lg transition-colors duration-300 ${isDarkMode ? 'text-purple-200' : 'text-purple-600'}`}>
                        Top 10 sequences ranked by average of Levenshtein and Hamming similarities
                      </p>
                    </div>

                    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
                      {generatedSequences.map((seq, index) => (
                        <div 
                          key={`seq-${index}`}
                          className={`p-5 rounded-xl border-2 transition-all duration-300 hover:shadow-xl flex flex-col ${
                            isDarkMode 
                              ? 'bg-slate-800/50 border-purple-600 hover:border-purple-500 shadow-lg' 
                              : 'bg-white border-purple-200 hover:border-purple-300 shadow-md'
                          }`}
                        >
                          <div className="space-y-3 flex-1 flex flex-col">
                            <div className="flex items-center gap-2 mb-2">
                              <div className={`w-7 h-7 rounded-full flex items-center justify-center text-white font-bold text-xs shrink-0 ${
                                index === 0 ? 'bg-yellow-500' : 
                                index === 1 ? 'bg-gray-400' : 
                                index === 2 ? 'bg-orange-500' : 'bg-blue-500'
                              }`}>
                                {index + 1}
                              </div>
                              <h4 className={`text-base font-semibold transition-colors duration-300 ${isDarkMode ? 'text-purple-100' : 'text-purple-800'}`}>
                                Sequence #{index + 1}
                              </h4>
                            </div>
                            
                            <div className={`font-mono text-xs p-2 rounded-lg transition-colors duration-300 overflow-auto max-h-24 ${
                              isDarkMode 
                                ? 'bg-slate-900 text-green-300 border border-slate-600' 
                                : 'bg-gray-900 text-green-300 border border-gray-700'
                            }`} style={{ wordBreak: 'break-all', lineHeight: '1.4' }}>
                              {seq.sequence}
                            </div>
                            
                            <div className={`text-center p-3 rounded-lg mt-auto ${isDarkMode ? 'bg-purple-900/50' : 'bg-purple-100'}`}>
                              <div className={`text-xl font-bold transition-colors duration-300 ${isDarkMode ? 'text-purple-200' : 'text-purple-800'}`}>
                                {(seq.average_probability * 100).toFixed(1)}%
                              </div>
                              <div className={`text-xs transition-colors duration-300 ${isDarkMode ? 'text-purple-300' : 'text-purple-600'}`}>
                                Avg Probability
                              </div>
                            </div>
                            
                            <div className="grid grid-cols-2 gap-2 mt-2">
                              <div className={`p-2 rounded-lg text-center ${isDarkMode ? 'bg-slate-600' : 'bg-gray-100'}`}>
                                <div className={`text-base font-bold transition-colors duration-300 ${isDarkMode ? 'text-blue-200' : 'text-gray-700'}`}>
                                  {(seq.levenshtein * 100).toFixed(1)}%
                                </div>
                                <div className={`text-xs font-medium transition-colors duration-300 ${isDarkMode ? 'text-blue-300' : 'text-gray-500'}`}>
                                  Levenshtein
                                </div>
                              </div>
                              <div className={`p-2 rounded-lg text-center ${isDarkMode ? 'bg-slate-600' : 'bg-gray-100'}`}>
                                <div className={`text-base font-bold transition-colors duration-300 ${isDarkMode ? 'text-green-200' : 'text-gray-700'}`}>
                                  {(seq.hamming * 100).toFixed(1)}%
                                </div>
                                <div className={`text-xs font-medium transition-colors duration-300 ${isDarkMode ? 'text-green-300' : 'text-gray-500'}`}>
                                  Hamming
                                </div>
                              </div>
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              )}
          </div>
          )}

          {/* SMILES Generation Tab Content */}
          {activeTab === 'smiles-generation' && (
          <div className={`p-8 rounded-xl mb-8 transition-all duration-300 ${isDarkMode ? 'bg-green-950/90 border border-green-600' : 'bg-green-50/90 border border-green-200'}`}>
            <label htmlFor="smilesSequence" className={`block text-xl font-semibold mb-4 transition-colors duration-300 ${isDarkMode ? 'text-green-100' : 'text-green-800'}`}>Protein Sequence (for SMILES)</label>
            <textarea
              id="smilesSequence"
              value={smilesSequence}
              onChange={(e) => setSmilesSequence(e.target.value)}
              placeholder="Enter protein sequence to generate SMILES..."
              className={`w-full h-32 px-6 py-4 border-2 rounded-xl focus:ring-2 focus:ring-green-500 focus:border-transparent resize-none text-base font-mono leading-relaxed transition-all duration-300 hover:border-gray-300 ${isDarkMode ? 'bg-slate-700 border-green-600 text-green-100 placeholder-green-300 focus:ring-green-400' : 'border-gray-200 text-gray-800 placeholder-gray-600'}`}
              disabled={smilesLoading}
            />
            <div className="mt-4 flex flex-col md:flex-row md:items-center md:gap-6 gap-3">
              <button
                onClick={handleGenerateSmiles}
                disabled={smilesLoading || !smilesSequence.trim()}
                className="w-full md:w-max bg-gradient-to-r from-green-500 to-teal-600 hover:from-green-600 hover:to-teal-700 disabled:from-gray-400 disabled:to-gray-500 text-white font-semibold py-4 px-8 rounded-xl transition-all duration-300 flex items-center justify-center gap-3 shadow-lg hover:shadow-xl transform hover:scale-[1.02] disabled:transform-none disabled:cursor-not-allowed text-lg"
              >
                {smilesLoading ? (<><Loader className="w-6 h-6 animate-spin" />Generating...</>) : (<><Activity className="w-6 h-6" />Generate SMILES</>)}
              </button>
              {smilesError && <div className="flex items-center gap-3 p-4 bg-red-50 border border-red-200 rounded-xl animate-fadeIn"><AlertCircle className="w-5 h-5 text-red-500 flex-shrink-0" /><p className="text-red-700">{smilesError}</p></div>}
            </div>
            {generatedSmiles && (
                  <div className="space-y-4 animate-fadeIn mt-6">
                    <div className="flex items-center gap-3 p-4 bg-green-50 border border-green-200 rounded-xl">
                      <CheckCircle className="w-5 h-5 text-green-500 flex-shrink-0" />
                      <div className="flex-1">
                        <p className="text-green-800 font-semibold text-lg">SMILES Generation Complete</p>
                        <p className="text-green-600 text-sm">Generated SMILES structure from protein sequence</p>
                      </div>
                      {smilesDevice && (
                        <div className={`px-3 py-1 rounded-lg text-xs font-semibold ${
                          smilesDevice.includes('GPU') ? 'bg-blue-100 text-blue-800' : 'bg-gray-100 text-gray-800'
                        }`}>
                          {smilesDevice}
                        </div>
                      )}
                    </div>

                    <div className={`p-6 rounded-xl border-2 transition-all duration-300 ${
                      isDarkMode 
                        ? 'bg-slate-800/50 border-green-600' 
                        : 'bg-white border-green-200'
                    }`}>
                      <h4 className={`text-lg font-semibold mb-3 transition-colors duration-300 ${isDarkMode ? 'text-green-100' : 'text-green-800'}`}>
                        Generated SMILES Structure
                      </h4>
                      <div className={`font-mono text-sm p-4 rounded-lg transition-colors duration-300 ${
                        isDarkMode 
                          ? 'bg-slate-900 text-green-300 border border-slate-600' 
                          : 'bg-gray-900 text-green-300 border border-gray-700'
                      }`}>
                        {generatedSmiles}
                      </div>
                    </div>
                  </div>
                )}
          </div>
          )}

          {/* AlphaFold2 Tab Content */}
          {activeTab === 'alphafold2' && (
            <div className={`p-8 rounded-xl mb-8 transition-all duration-300 ${isDarkMode ? 'bg-indigo-950/90 border border-indigo-600' : 'bg-indigo-50/90 border border-indigo-200'}`}>
              <label htmlFor="alphafold-sequence" className={`block text-xl font-semibold mb-4 transition-colors duration-300 ${isDarkMode ? 'text-indigo-100' : 'text-indigo-800'}`}>
                Protein Sequence
              </label>
              <textarea
                id="alphafold-sequence"
                value={alphafoldSequence}
                onChange={(e) => setAlphafoldSequence(e.target.value)}
                placeholder="Enter protein sequence to predict 3D structure..."
                className={`w-full h-40 px-6 py-4 border-2 rounded-xl focus:ring-2 focus:ring-indigo-500 focus:border-transparent resize-none text-base font-mono leading-relaxed transition-all duration-300 ${isDarkMode ? 'bg-slate-700 border-indigo-600 text-indigo-100 placeholder-indigo-300 focus:ring-indigo-400' : 'border-gray-200 text-gray-800 placeholder-gray-600'}`}
                disabled={alphafoldLoading}
              />
              <div className="mt-4 flex flex-col md:flex-row md:items-center md:gap-6 gap-3">
                <button
                  onClick={handleAlphaFold2Predict}
                  disabled={alphafoldLoading || !alphafoldSequence.trim()}
                  className="w-full md:w-max bg-gradient-to-r from-indigo-500 to-blue-600 hover:from-indigo-600 hover:to-blue-700 disabled:from-gray-400 disabled:to-gray-500 text-white font-semibold py-4 px-8 rounded-xl transition-all duration-300 flex items-center justify-center gap-3 shadow-lg hover:shadow-xl transform hover:scale-[1.02] disabled:transform-none disabled:cursor-not-allowed text-lg"
                >
                  {alphafoldLoading ? (
                    <>
                      <Loader className="w-6 h-6 animate-spin" />
                      Predicting Structure...
                    </>
                  ) : (
                    <>
                      <Dna className="w-6 h-6" />
                      Predict Structure
                    </>
                  )}
                </button>
                {alphafoldError && (
                  <div className="flex items-center gap-3 p-4 bg-red-50 border border-red-200 rounded-xl animate-fadeIn">
                    <AlertCircle className="w-5 h-5 text-red-500 flex-shrink-0" />
                    <p className="text-red-700">{alphafoldError}</p>
                  </div>
                )}
              </div>
              {alphafoldResult && alphafoldResult.success && (
                <div className="space-y-6 animate-fadeIn mt-6">
                  <div className="flex items-center gap-3 p-5 bg-green-50 border border-green-200 rounded-xl">
                    <CheckCircle className="w-6 h-6 text-green-500 flex-shrink-0" />
                    <div className="flex-1">
                      <p className="text-green-800 font-semibold text-lg">Structure Prediction Complete</p>
                      <p className="text-green-600 text-sm">AlphaFold2 has generated the 3D structure</p>
                    </div>
                    {alphafoldResult?.device_type && (
                      <div className={`px-3 py-1 rounded-lg text-xs font-semibold ${
                        alphafoldResult.device_type.includes('GPU') ? 'bg-blue-100 text-blue-800' : 'bg-gray-100 text-gray-800'
                      }`}>
                        {alphafoldResult.device_type}
                      </div>
                    )}
                  </div>
                  <div className={`rounded-xl p-8 space-y-6 transition-all duration-300 ${isDarkMode ? 'bg-slate-700/50 border border-indigo-600' : 'bg-indigo-50'}`}>
                    <div className="flex items-center justify-between">
                      <h3 className={`text-2xl font-bold transition-colors duration-300 ${isDarkMode ? 'text-indigo-100' : 'text-indigo-800'}`}>
                        Prediction Results
                      </h3>
                      {alphafoldResult.result.pdb_content && (
                        <button
                          onClick={downloadAlphaFoldPDB}
                          className={`px-4 py-2 rounded-lg flex items-center gap-2 transition-colors ${isDarkMode ? 'bg-indigo-600 hover:bg-indigo-700 text-white' : 'bg-indigo-600 hover:bg-indigo-700 text-white'}`}
                        >
                          <Download className="w-4 h-4" />
                          Download PDB
                        </button>
                      )}
                    </div>
                    {alphafoldResult.result.plddt_score && (
                      <div className={`rounded-lg p-4 border ${isDarkMode ? 'bg-slate-800/50 border-indigo-600' : 'bg-white/50 border-indigo-200'}`}>
                        <h4 className={`text-lg font-semibold mb-2 transition-colors duration-300 ${isDarkMode ? 'text-indigo-100' : 'text-indigo-800'}`}>
                          Confidence Score (pLDDT)
                        </h4>
                        <div className={`text-3xl font-bold mb-2 ${isDarkMode ? 'text-indigo-300' : 'text-indigo-600'}`}>
                          {typeof alphafoldResult.result.plddt_score === 'number' 
                            ? alphafoldResult.result.plddt_score.toFixed(2)
                            : alphafoldResult.result.plddt_score}
                        </div>
                        <p className={`text-sm ${isDarkMode ? 'text-indigo-200' : 'text-indigo-600'}`}>
                          {typeof alphafoldResult.result.plddt_score === 'number' && (
                            <>
                              {alphafoldResult.result.plddt_score >= 90 ? 'Very high confidence' :
                               alphafoldResult.result.plddt_score >= 70 ? 'Confident' :
                               alphafoldResult.result.plddt_score >= 50 ? 'Low confidence' :
                               'Very low confidence'}
                              {' '}(pLDDT: 0-100 scale)
                            </>
                          )}
                        </p>
                      </div>
                    )}
                    {alphafoldResult.result.pdb_content && (
                      <div className={`rounded-lg p-4 border ${isDarkMode ? 'bg-slate-900 border-slate-600' : 'bg-gray-900'}`}>
                        <h4 className={`text-lg font-semibold mb-2 transition-colors duration-300 ${isDarkMode ? 'text-indigo-100' : 'text-indigo-800'}`}>
                          PDB File Content (Preview)
                        </h4>
                        <pre className="text-green-400 text-xs overflow-auto max-h-64 font-mono">
                          {alphafoldResult.result.pdb_content.substring(0, 1000)}
                          {alphafoldResult.result.pdb_content.length > 1000 && '...'}
                        </pre>
                      </div>
                    )}
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Docking Tab Content */}
          {activeTab === 'docking' && (
            <div className={`p-8 rounded-xl mb-8 transition-all duration-300 ${isDarkMode ? 'bg-teal-950/90 border border-teal-600' : 'bg-teal-50/90 border border-teal-200'}`}>
              <div className="space-y-6">
                <div>
                  <label htmlFor="docking-smiles" className={`block text-xl font-semibold mb-4 transition-colors duration-300 ${isDarkMode ? 'text-teal-100' : 'text-teal-800'}`}>
                    SMILES String (Drug Molecule)
                  </label>
                  <input
                    id="docking-smiles"
                    type="text"
                    value={dockingSmiles}
                    onChange={(e) => setDockingSmiles(e.target.value)}
                    placeholder="Enter SMILES string (e.g., C[C@H](N)C(=O)O) or use generated SMILES from above"
                    className={`w-full px-6 py-4 border-2 rounded-xl focus:ring-2 focus:ring-teal-500 focus:border-transparent text-base font-mono transition-all duration-300 ${isDarkMode ? 'bg-slate-700 border-teal-600 text-teal-100 placeholder-teal-300 focus:ring-teal-400' : 'border-gray-200 text-gray-800 placeholder-gray-600'}`}
                    disabled={dockingLoading}
                  />
                </div>
                <div>
                  <label className="flex items-center gap-3 mb-4">
                    <input
                      type="checkbox"
                      checked={useProteinSequence}
                      onChange={(e) => setUseProteinSequence(e.target.checked)}
                      className="w-4 h-4 rounded"
                    />
                    <span className={isDarkMode ? 'text-teal-100' : 'text-teal-800'}>Use Protein Sequence (otherwise use PDB file from AlphaFold2)</span>
                  </label>
                </div>
                {useProteinSequence ? (
                  <div>
                    <label htmlFor="docking-protein-sequence" className={`block text-xl font-semibold mb-4 transition-colors duration-300 ${isDarkMode ? 'text-teal-100' : 'text-teal-800'}`}>
                      Protein Sequence
                    </label>
                    <textarea
                      id="docking-protein-sequence"
                      value={dockingProteinSequence}
                      onChange={(e) => setDockingProteinSequence(e.target.value)}
                      placeholder="Enter protein sequence"
                      className={`w-full h-32 px-6 py-4 border-2 rounded-xl focus:ring-2 focus:ring-teal-500 focus:border-transparent resize-none text-base font-mono leading-relaxed transition-all duration-300 ${isDarkMode ? 'bg-slate-700 border-teal-600 text-teal-100 placeholder-teal-300 focus:ring-teal-400' : 'border-gray-200 text-gray-800 placeholder-gray-600'}`}
                      disabled={dockingLoading}
                    />
                  </div>
                ) : (
                  <div>
                    <label htmlFor="docking-protein-pdb" className={`block text-xl font-semibold mb-4 transition-colors duration-300 ${isDarkMode ? 'text-teal-100' : 'text-teal-800'}`}>
                      Protein PDB File Content
                    </label>
                    <textarea
                      id="docking-protein-pdb"
                      value={dockingProteinPDB}
                      onChange={(e) => setDockingProteinPDB(e.target.value)}
                      placeholder="Paste PDB file content here (from AlphaFold2 prediction result)"
                      className={`w-full h-40 px-6 py-4 border-2 rounded-xl focus:ring-2 focus:ring-teal-500 focus:border-transparent resize-none text-base font-mono leading-relaxed transition-all duration-300 ${isDarkMode ? 'bg-slate-700 border-teal-600 text-teal-100 placeholder-teal-300 focus:ring-teal-400' : 'border-gray-200 text-gray-800 placeholder-gray-600'}`}
                      disabled={dockingLoading}
                    />
                    <p className={`text-sm mt-2 ${isDarkMode ? 'text-teal-200' : 'text-teal-600'}`}>
                      Tip: Use the PDB file from AlphaFold2 prediction results
                    </p>
                  </div>
                )}
                <div className="flex flex-col md:flex-row md:items-center md:gap-6 gap-3">
                  <button
                    onClick={handleDocking}
                    disabled={dockingLoading || !dockingSmiles.trim()}
                    className="w-full md:w-max bg-gradient-to-r from-teal-500 to-cyan-600 hover:from-teal-600 hover:to-cyan-700 disabled:from-gray-400 disabled:to-gray-500 text-white font-semibold py-4 px-8 rounded-xl transition-all duration-300 flex items-center justify-center gap-3 shadow-lg hover:shadow-xl transform hover:scale-[1.02] disabled:transform-none disabled:cursor-not-allowed text-lg"
                  >
                    {dockingLoading ? (
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
                  {dockingError && (
                    <div className="flex items-center gap-3 p-4 bg-red-50 border border-red-200 rounded-xl animate-fadeIn">
                      <AlertCircle className="w-5 h-5 text-red-500 flex-shrink-0" />
                      <p className="text-red-700">{dockingError}</p>
                    </div>
                  )}
                </div>
                {dockingResult && dockingResult.success && (
                  <div className="space-y-6 animate-fadeIn mt-6">
                    <div className="flex items-center gap-3 p-5 bg-green-50 border border-green-200 rounded-xl">
                      <CheckCircle className="w-6 h-6 text-green-500 flex-shrink-0" />
                      <div className="flex-1">
                        <p className="text-green-800 font-semibold text-lg">Docking Completed</p>
                        <p className="text-green-600 text-sm">Binding affinity scores calculated</p>
                      </div>
                      {dockingResult?.device_type && (
                        <div className={`px-3 py-1 rounded-lg text-xs font-semibold ${
                          dockingResult.device_type.includes('GPU') ? 'bg-blue-100 text-blue-800' : 'bg-gray-100 text-gray-800'
                        }`}>
                          {dockingResult.device_type}
                        </div>
                      )}
                    </div>
                    <div className={`rounded-xl p-8 space-y-6 transition-all duration-300 ${isDarkMode ? 'bg-slate-700/50 border border-teal-600' : 'bg-teal-50'}`}>
                      <div className="flex items-center justify-between">
                        <h3 className={`text-2xl font-bold transition-colors duration-300 ${isDarkMode ? 'text-teal-100' : 'text-teal-800'}`}>
                          Docking Results
                        </h3>
                        {dockingResult.result.pdbqt_content && (
                          <button
                            onClick={downloadDockingPDBQT}
                            className={`px-4 py-2 rounded-lg flex items-center gap-2 transition-colors ${isDarkMode ? 'bg-teal-600 hover:bg-teal-700 text-white' : 'bg-teal-600 hover:bg-teal-700 text-white'}`}
                          >
                            <Download className="w-4 h-4" />
                            Download PDBQT
                          </button>
                        )}
                      </div>
                      {dockingResult.result.best_affinity !== undefined && (
                        <div className={`rounded-lg p-6 border ${isDarkMode ? 'bg-slate-800/50 border-teal-600' : 'bg-white/50 border-teal-200'}`}>
                          <h4 className={`text-lg font-semibold mb-4 transition-colors duration-300 ${isDarkMode ? 'text-teal-100' : 'text-teal-800'}`}>
                            Binding Affinity
                          </h4>
                          <div className={`text-4xl font-bold mb-2 ${isDarkMode ? 'text-teal-300' : 'text-teal-600'}`}>
                            {dockingResult.result.best_affinity.toFixed(2)} kcal/mol
                          </div>
                          <p className={isDarkMode ? 'text-teal-200 text-sm' : 'text-teal-600 text-sm'}>
                            {dockingResult.result.best_affinity < -7 ? 'Strong binding' :
                             dockingResult.result.best_affinity < -5 ? 'Moderate binding' :
                             'Weak binding'}
                          </p>
                        </div>
                      )}
                      {dockingResult.result.affinities && dockingResult.result.affinities.length > 0 && (
                        <div className={`rounded-lg p-4 border ${isDarkMode ? 'bg-slate-800/50 border-teal-600' : 'bg-white/50 border-teal-200'}`}>
                          <h4 className={`text-lg font-semibold mb-4 transition-colors duration-300 ${isDarkMode ? 'text-teal-100' : 'text-teal-800'}`}>
                            All Pose Affinities
                          </h4>
                          <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
                            {dockingResult.result.affinities.map((affinity: number, idx: number) => (
                              <div key={idx} className={`text-center p-3 rounded-lg ${isDarkMode ? 'bg-slate-700/50' : 'bg-white/50'}`}>
                                <div className={`text-xs mb-1 ${isDarkMode ? 'text-teal-200' : 'text-teal-600'}`}>Pose {idx + 1}</div>
                                <div className={`text-xl font-bold ${isDarkMode ? 'text-teal-300' : 'text-teal-600'}`}>
                                  {affinity.toFixed(2)}
                                </div>
                                <div className={`text-xs ${isDarkMode ? 'text-teal-300' : 'text-teal-600'}`}>kcal/mol</div>
                              </div>
                            ))}
                          </div>
                        </div>
                      )}
                      {(dockingResult.result.visualization_data || dockingResult.result.visualization_url || dockingResult.result.visualization_image) && (
                        <div className={`rounded-lg p-4 border ${isDarkMode ? 'bg-slate-800/50 border-teal-600' : 'bg-white/50 border-teal-200'}`}>
                          <h4 className={`text-lg font-semibold mb-4 transition-colors duration-300 ${isDarkMode ? 'text-teal-100' : 'text-teal-800'}`}>
                            3D Molecular Visualization
                          </h4>
                          {dockingResult.result.visualization_data && (dockingResult.result.visualization_data.protein_pdb || dockingResult.result.protein_pdb) && dockingResult.result.visualization_data.ligand_pdbqt ? (
                            <div className="relative" style={{ zIndex: 0 }}>
                              <Molecule3DViewer
                                proteinPdb={dockingResult.result.visualization_data.protein_pdb || dockingResult.result.protein_pdb || ''}
                                ligandPdbqt={dockingResult.result.visualization_data.ligand_pdbqt || dockingResult.result.pdbqt_content || ''}
                              />
                            </div>
                          ) : dockingResult.result.visualization_url ? (
                            <iframe
                              src={dockingResult.result.visualization_url}
                              className="w-full h-96 rounded-lg border border-white/20"
                              title="Docking Visualization"
                            />
                          ) : dockingResult.result.visualization_image ? (
                            <div className="flex justify-center items-center bg-white rounded-lg p-4">
                              <img
                                src={dockingResult.result.visualization_image}
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
          )}
          
          {/* Pipeline Tab Content */}
          {activeTab === 'pipeline' && (
            <div className={`p-8 rounded-xl mb-8 transition-all duration-300 ${isDarkMode ? 'bg-orange-950/90 border border-orange-600' : 'bg-orange-50/90 border border-orange-200'}`}>
              <div className="space-y-6">
                <div className="mb-6">
                  <h2 className={`text-2xl font-bold mb-2 transition-colors duration-300 ${isDarkMode ? 'text-orange-100' : 'text-orange-800'}`}>
                    Complete Pipeline: AlphaFold2 → Docking
                  </h2>
                  <p className={`text-sm transition-colors duration-300 ${isDarkMode ? 'text-orange-200' : 'text-orange-700'}`}>
                    Automatically run AlphaFold2 prediction, extract rank 1 PDB from ZIP, and perform docking in one step.
                    This process can take 15-20 minutes. Rank 1 PDB is automatically extracted from AlphaFold2 ZIP output.
                  </p>
                </div>
                
                <div>
                  <label htmlFor="pipeline-sequence" className={`block text-xl font-semibold mb-4 transition-colors duration-300 ${isDarkMode ? 'text-orange-100' : 'text-orange-800'}`}>
                    Protein Sequence
                  </label>
                  <textarea
                    id="pipeline-sequence"
                    value={pipelineSequence}
                    onChange={(e) => setPipelineSequence(e.target.value)}
                    placeholder="Enter protein sequence for AlphaFold2 prediction..."
                    className={`w-full h-32 px-6 py-4 border-2 rounded-xl focus:ring-2 focus:ring-orange-500 focus:border-transparent resize-none text-base font-mono leading-relaxed transition-all duration-300 ${isDarkMode ? 'bg-slate-700 border-orange-600 text-orange-100 placeholder-orange-300 focus:ring-orange-400' : 'border-gray-200 text-gray-800 placeholder-gray-600'}`}
                    disabled={pipelineLoading}
                  />
                </div>
                
                <div>
                  <label htmlFor="pipeline-smiles" className={`block text-xl font-semibold mb-4 transition-colors duration-300 ${isDarkMode ? 'text-orange-100' : 'text-orange-800'}`}>
                    SMILES String (Drug Molecule)
                  </label>
                  <input
                    id="pipeline-smiles"
                    type="text"
                    value={pipelineSmiles}
                    onChange={(e) => setPipelineSmiles(e.target.value)}
                    placeholder="Enter SMILES string (e.g., C[C@H](N)C(=O)O) or use generated SMILES from above"
                    className={`w-full px-6 py-4 border-2 rounded-xl focus:ring-2 focus:ring-orange-500 focus:border-transparent text-base font-mono transition-all duration-300 ${isDarkMode ? 'bg-slate-700 border-orange-600 text-orange-100 placeholder-orange-300 focus:ring-orange-400' : 'border-gray-200 text-gray-800 placeholder-gray-600'}`}
                    disabled={pipelineLoading}
                  />
                </div>
                
                <div className="flex items-center gap-4">
                  <button
                    onClick={handlePipelineRun}
                    disabled={pipelineLoading || !pipelineSequence.trim() || !pipelineSmiles.trim()}
                    className={`px-8 py-4 rounded-xl transition-all duration-300 flex items-center justify-center gap-3 shadow-lg hover:shadow-xl transform hover:scale-[1.02] disabled:transform-none disabled:cursor-not-allowed text-lg font-semibold ${
                      pipelineLoading
                        ? 'bg-gray-400 text-gray-600'
                        : 'bg-gradient-to-r from-orange-500 to-red-600 hover:from-orange-600 hover:to-red-700 text-white'
                    }`}
                  >
                    {pipelineLoading ? (
                      <>
                        <Loader className="w-6 h-6 animate-spin" />
                        {pipelineStep || 'Running Pipeline...'}
                      </>
                    ) : (
                      <>
                        <FlaskConical className="w-6 h-6" />
                        Run Complete Pipeline
                      </>
                    )}
                  </button>
                  {pipelineError && (
                    <div className="flex items-center gap-3 p-4 bg-red-50 border border-red-200 rounded-xl animate-fadeIn">
                      <AlertCircle className="w-5 h-5 text-red-500 flex-shrink-0" />
                      <p className="text-red-700">{pipelineError}</p>
                    </div>
                  )}
                </div>
              </div>
              
              {pipelineResult && pipelineResult.success && (
                <div className="space-y-6 animate-fadeIn mt-8">
                  <div className="flex items-center gap-3 p-5 bg-green-50 border border-green-200 rounded-xl">
                    <CheckCircle className="w-6 h-6 text-green-500 flex-shrink-0" />
                    <div className="flex-1">
                      <p className="text-green-800 font-semibold text-lg">Pipeline Completed Successfully!</p>
                      <p className="text-green-600 text-sm">AlphaFold2 prediction and docking completed</p>
                    </div>
                    {pipelineResult.device_type && (
                      <div className={`px-3 py-1 rounded-lg text-xs font-semibold ${
                        pipelineResult.device_type.includes('GPU') ? 'bg-blue-100 text-blue-800' : 'bg-gray-100 text-gray-800'
                      }`}>
                        {pipelineResult.device_type}
                      </div>
                    )}
                  </div>
                  
                  {/* AlphaFold2 Results */}
                  {pipelineResult.alphafold2 && (
                    <div className={`rounded-xl p-8 space-y-4 transition-all duration-300 ${isDarkMode ? 'bg-slate-700/50 border border-indigo-600' : 'bg-indigo-50'}`}>
                      <h3 className={`text-xl font-bold transition-colors duration-300 ${isDarkMode ? 'text-indigo-100' : 'text-indigo-800'}`}>
                        AlphaFold2 Results (Rank 1 PDB)
                      </h3>
                      <div className="flex items-center justify-between">
                        <div>
                          <p className={`text-sm transition-colors duration-300 ${isDarkMode ? 'text-indigo-200' : 'text-indigo-700'}`}>
                            pLDDT Score: <span className="font-bold">{pipelineResult.alphafold2.plddt_score}</span>
                          </p>
                          <p className={`text-xs transition-colors duration-300 ${isDarkMode ? 'text-indigo-300' : 'text-indigo-600'}`}>
                            Rank 1 File: {pipelineResult.alphafold2.rank1_file}
                          </p>
                        </div>
                        <button
                          onClick={() => {
                            const blob = new Blob([pipelineResult.alphafold2.pdb_content], { type: 'text/plain' });
                            const url = URL.createObjectURL(blob);
                            const a = document.createElement('a');
                            a.href = url;
                            a.download = `alphafold2_rank1_${pipelineResult.protein_sequence.substring(0, 10)}.pdb`;
                            document.body.appendChild(a);
                            a.click();
                            document.body.removeChild(a);
                            URL.revokeObjectURL(url);
                          }}
                          className="px-4 py-2 bg-indigo-600 hover:bg-indigo-700 text-white rounded-lg flex items-center gap-2 transition-colors"
                        >
                          <Download className="w-4 h-4" />
                          Download Rank 1 PDB
                        </button>
                      </div>
                    </div>
                  )}
                  
                  {/* Docking Results */}
                  {pipelineResult.docking && (
                    <div className={`rounded-xl p-8 space-y-6 transition-all duration-300 ${isDarkMode ? 'bg-slate-700/50 border border-teal-600' : 'bg-teal-50'}`}>
                      <h3 className={`text-xl font-bold transition-colors duration-300 ${isDarkMode ? 'text-teal-100' : 'text-teal-800'}`}>
                        Docking Results
                      </h3>
                      {pipelineResult.docking.best_affinity !== undefined && (
                        <div className={`rounded-lg p-6 border ${isDarkMode ? 'bg-white/5 border-white/10' : 'bg-white/5 border-white/10'}`}>
                          <h4 className={`text-lg font-semibold mb-4 transition-colors duration-300 ${isDarkMode ? 'text-teal-100' : 'text-teal-800'}`}>
                            Best Binding Affinity
                          </h4>
                          <div className="text-4xl font-bold text-green-300 mb-2">
                            {pipelineResult.docking.best_affinity.toFixed(2)} kcal/mol
                          </div>
                          <p className={`text-sm transition-colors duration-300 ${isDarkMode ? 'text-teal-200' : 'text-teal-700'}`}>
                            {pipelineResult.docking.best_affinity < -7 ? 'Strong binding' :
                             pipelineResult.docking.best_affinity < -5 ? 'Moderate binding' :
                             'Weak binding'}
                          </p>
                        </div>
                      )}
                      {pipelineResult.docking.affinities && pipelineResult.docking.affinities.length > 0 && (
                        <div className={`rounded-lg p-4 border ${isDarkMode ? 'bg-white/5 border-white/10' : 'bg-white/5 border-white/10'}`}>
                          <h4 className={`text-lg font-semibold mb-4 transition-colors duration-300 ${isDarkMode ? 'text-teal-100' : 'text-teal-800'}`}>
                            All Pose Affinities
                          </h4>
                          <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
                            {pipelineResult.docking.affinities.map((affinity: number, idx: number) => (
                              <div key={idx} className={`text-center p-3 rounded-lg ${isDarkMode ? 'bg-white/5' : 'bg-white/5'}`}>
                                <div className={`text-xs mb-1 transition-colors duration-300 ${isDarkMode ? 'text-teal-200' : 'text-teal-700'}`}>
                                  Pose {idx + 1}
                                </div>
                                <div className="text-xl font-bold text-green-300">{affinity.toFixed(2)}</div>
                                <div className={`text-xs transition-colors duration-300 ${isDarkMode ? 'text-teal-300' : 'text-teal-600'}`}>
                                  kcal/mol
                                </div>
                              </div>
                            ))}
                          </div>
                        </div>
                      )}
                      {pipelineResult.docking.pdbqt_content && (
                        <button
                          onClick={() => {
                            const blob = new Blob([pipelineResult.docking.pdbqt_content], { type: 'text/plain' });
                            const url = URL.createObjectURL(blob);
                            const a = document.createElement('a');
                            a.href = url;
                            a.download = `docked_poses_${pipelineResult.smiles.substring(0, 10).replace(/[^a-zA-Z0-9]/g, '_')}.pdbqt`;
                            document.body.appendChild(a);
                            a.click();
                            document.body.removeChild(a);
                            URL.revokeObjectURL(url);
                          }}
                          className="px-4 py-2 bg-teal-600 hover:bg-teal-700 text-white rounded-lg flex items-center gap-2 transition-colors"
                        >
                          <Download className="w-4 h-4" />
                          Download Docked Poses (PDBQT)
                        </button>
                      )}
                      {(pipelineResult.docking.visualization_data || pipelineResult.docking.visualization_url || pipelineResult.docking.visualization_image) && (
                        <div className={`rounded-lg p-4 border ${isDarkMode ? 'bg-white/5 border-white/10' : 'bg-white/5 border-white/10'}`}>
                          <h4 className={`text-lg font-semibold mb-4 transition-colors duration-300 ${isDarkMode ? 'text-teal-100' : 'text-teal-800'}`}>
                            3D Molecular Visualization
                          </h4>
                          {pipelineResult.docking.visualization_data && pipelineResult.docking.visualization_data.protein_pdb && pipelineResult.docking.visualization_data.ligand_pdbqt ? (
                            <div className="relative" style={{ zIndex: 0 }}>
                              <Molecule3DViewer
                                proteinPdb={pipelineResult.docking.visualization_data.protein_pdb}
                                ligandPdbqt={pipelineResult.docking.visualization_data.ligand_pdbqt}
                              />
                            </div>
                          ) : pipelineResult.docking.visualization_url ? (
                            <iframe
                              src={pipelineResult.docking.visualization_url}
                              className="w-full h-96 rounded-lg border border-white/20"
                              title="Docking Visualization"
                            />
                          ) : pipelineResult.docking.visualization_image ? (
                            <div className="flex justify-center items-center bg-white rounded-lg p-4">
                              <img
                                src={pipelineResult.docking.visualization_image}
                                alt="Ligand Molecular Structure"
                                className="max-w-full h-auto rounded-lg shadow-lg"
                                style={{ maxHeight: '600px' }}
                              />
                            </div>
                          ) : null}
                        </div>
                      )}
                    </div>
                  )}
                </div>
              )}
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="mt-12 text-center">
          <p className={`text-lg transition-colors duration-300 ${isDarkMode ? 'text-blue-200' : 'text-blue-200'}`}>
            Powered by advanced machine learning algorithms
          </p>
        </div>
      </div>
    </div>
  );
}

