import { useState } from 'react';
import { Loader, CheckCircle, AlertCircle, Dna, Download } from 'lucide-react';

interface AlphaFold2Result {
  success: boolean;
  sequence: string;
  result: {
    pdb_content?: string;
    pdb_url?: string;
    confidence_scores?: Record<string, number>;
    plddt_score?: number;
    error?: string;
  };
  error?: string;
}

export default function AlphaFold2Page() {
  const [sequence, setSequence] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<AlphaFold2Result | null>(null);
  const [error, setError] = useState<string | null>(null);

  const validateSequence = (seq: string): boolean => {
    const validAminoAcids = /^[ACDEFGHIKLMNPQRSTVWY]+$/i;
    return validAminoAcids.test(seq.trim());
  };

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
      setError('Invalid amino acid sequence. Please use single-letter amino acid codes.');
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await fetch('http://localhost:5001/alphafold2/predict', {
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

      const data: AlphaFold2Result = await response.json();

      if (data.success) {
        setResult(data);
      } else {
        setError(data.error || 'AlphaFold2 prediction failed');
      }
    } catch (err) {
      console.error('AlphaFold2 error:', err);
      setError('Unable to connect to the server. Please ensure the Flask backend and Colab notebook (via ngrok) are running.');
    } finally {
      setLoading(false);
    }
  };

  const downloadPDB = () => {
    if (result?.result?.pdb_content) {
      const blob = new Blob([result.result.pdb_content], { type: 'text/plain' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `alphafold2_prediction_${result.sequence.substring(0, 10)}.pdb`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-900 via-indigo-900 to-purple-900 py-16">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <h1 className="text-4xl font-bold text-white mb-4">AlphaFold2 Structure Prediction</h1>
        <p className="text-blue-200 mb-12 text-lg">
          Predict 3D protein structures using AlphaFold2 powered by ColabFold
        </p>

        <div className="bg-white/10 backdrop-blur-sm rounded-xl border border-white/20 p-8 shadow-sm mb-8">
          <label htmlFor="alphafold-sequence" className="block text-xl font-semibold mb-4 text-white">
            Protein Sequence
          </label>
          <textarea
            id="alphafold-sequence"
            value={sequence}
            onChange={(e) => setSequence(e.target.value)}
            placeholder="Enter your protein sequence here (e.g., MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWAGIKATEAAVSEEFGLAPFLPDQIHFVHSQELLSRYPDLDAKGRERAIAKDLGAVFLVGIGGKLSDGHRHDVRAPDYDDWQTSTSTSLPRADLQLFVDGVRQLEWLSQRLQQPQQKSAFAVQEDFNRSWFRPGHRRNKVFDLPIGVLKSSAQNLMNQEDVHSKQAPGTILKSQGMQVFVLEELDKTLFTLGFHKPAIVQHASSAKDLGPLLDGIWKTTTTKQAAKCLQKNLPSFLGVTSSEFRYLMNSQTRLPDNYLPLLPAIIDRFDNTLPLTGQAQIIFRRFLPLQGKEFQ)"
            className="w-full h-40 px-6 py-4 border-2 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none text-base font-mono leading-relaxed transition-all duration-300 bg-white/10 border-white/30 text-white placeholder-white/50 focus:ring-blue-400"
            disabled={loading}
          />
          <div className="mt-4 flex items-center gap-4">
            <button
              onClick={handlePredict}
              disabled={loading || !sequence.trim()}
              className="px-8 py-4 bg-gradient-to-r from-blue-500 to-indigo-600 hover:from-blue-600 hover:to-indigo-700 disabled:from-gray-400 disabled:to-gray-500 text-white font-semibold rounded-xl transition-all duration-300 flex items-center justify-center gap-3 shadow-lg hover:shadow-xl transform hover:scale-[1.02] disabled:transform-none disabled:cursor-not-allowed text-lg"
            >
              {loading ? (
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
                <p className="text-green-800 font-semibold text-lg">Structure Prediction Complete</p>
                <p className="text-green-600 text-sm">AlphaFold2 has generated the 3D structure</p>
              </div>
            </div>

            <div className="bg-white/10 backdrop-blur-sm rounded-xl border border-white/20 p-8 shadow-sm space-y-6">
              <div className="flex items-center justify-between">
                <h3 className="text-2xl font-bold text-white">Prediction Results</h3>
                {result.result.pdb_content && (
                  <button
                    onClick={downloadPDB}
                    className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg flex items-center gap-2 transition-colors"
                  >
                    <Download className="w-4 h-4" />
                    Download PDB
                  </button>
                )}
              </div>

              {result.result.plddt_score && (
                <div className="bg-white/5 rounded-lg p-4 border border-white/10">
                  <h4 className="text-lg font-semibold text-white mb-2">Confidence Score (pLDDT)</h4>
                  <div className="text-3xl font-bold text-blue-300">{result.result.plddt_score.toFixed(2)}</div>
                  <p className="text-blue-200 text-sm mt-2">Higher scores indicate better confidence</p>
                </div>
              )}

              {result.result.confidence_scores && (
                <div className="bg-white/5 rounded-lg p-4 border border-white/10">
                  <h4 className="text-lg font-semibold text-white mb-4">Per-Residue Confidence</h4>
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                    {Object.entries(result.result.confidence_scores).map(([residue, score]) => (
                      <div key={residue} className="text-center">
                        <div className="text-sm text-blue-200">{residue}</div>
                        <div className="text-xl font-bold text-white">{score.toFixed(2)}</div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {result.result.pdb_url && (
                <div className="bg-white/5 rounded-lg p-4 border border-white/10">
                  <h4 className="text-lg font-semibold text-white mb-2">3D Structure Viewer</h4>
                  <iframe
                    src={result.result.pdb_url}
                    className="w-full h-96 rounded-lg border border-white/20"
                    title="AlphaFold2 Structure Viewer"
                  />
                </div>
              )}

              {result.result.pdb_content && (
                <div className="bg-gray-900 rounded-lg p-4 border border-white/10">
                  <h4 className="text-lg font-semibold text-white mb-2">PDB File Content (Preview)</h4>
                  <pre className="text-green-400 text-xs overflow-auto max-h-64 font-mono">
                    {result.result.pdb_content.substring(0, 1000)}
                    {result.result.pdb_content.length > 1000 && '...'}
                  </pre>
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}


