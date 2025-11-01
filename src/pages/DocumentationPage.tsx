export default function DocumentationPage() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-900 via-indigo-900 to-purple-900 py-16">
      <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
        <h1 className="text-4xl font-bold text-white mb-8">Documentation</h1>

        <div className="bg-white/10 backdrop-blur-sm rounded-xl border border-white/20 p-8 shadow-sm space-y-8">
          <section>
            <h2 className="text-2xl font-semibold text-white mb-4">Getting Started</h2>
            <p className="text-blue-200 leading-relaxed mb-4">
              Welcome to the Insulin Drug Synthesis platform. This guide will help you get started 
              with analyzing protein sequences, generating variants, and converting to molecular structures.
            </p>
            <div className="bg-white/5 rounded-lg p-4 mb-4 border border-white/10">
              <h3 className="font-semibold text-white mb-2">Quick Start</h3>
              <ol className="list-decimal list-inside space-y-2 text-blue-200">
                <li>Sign up for a free account</li>
                <li>Navigate to the Dashboard</li>
                <li>Enter your protein sequence</li>
                <li>Select the analysis type (Classification, Generation, or SMILES)</li>
                <li>View your results</li>
              </ol>
            </div>
          </section>

          <section>
            <h2 className="text-2xl font-semibold text-white mb-4">API Endpoints</h2>
            <div className="space-y-4">
              <div className="bg-white/5 rounded-lg p-4 border border-white/10">
                <code className="text-sm text-blue-300">POST /predict</code>
                <p className="text-blue-200 text-sm mt-2">Classify a protein sequence</p>
              </div>
              <div className="bg-white/5 rounded-lg p-4 border border-white/10">
                <code className="text-sm text-purple-300">POST /generate-sequences</code>
                <p className="text-blue-200 text-sm mt-2">Generate protein sequence variants</p>
              </div>
              <div className="bg-white/5 rounded-lg p-4 border border-white/10">
                <code className="text-sm text-green-300">POST /generate-smiles</code>
                <p className="text-blue-200 text-sm mt-2">Convert sequence to SMILES format</p>
              </div>
            </div>
          </section>

          <section>
            <h2 className="text-2xl font-semibold text-white mb-4">Sequence Format</h2>
            <p className="text-blue-200 leading-relaxed mb-4">
              Protein sequences should be in single-letter amino acid code format. Valid amino acids 
              include: A, C, D, E, F, G, H, I, K, L, M, N, P, Q, R, S, T, V, W, Y.
            </p>
            <div className="bg-gray-900 rounded-lg p-4">
              <code className="text-green-400 text-sm">MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWAGIKATEAAVSEEFGLAPFLPDQIHFVHSQELLSRYPDLDAKGRERAIAKDLGAVFLVGIGGKLSDGHRHDVRAPDYDDWQTSTSTSLPRADLQLFVDGVRQLEWLSQRLQQPQQKSAFAVQEDFNRSWFRPGHRRNKVFDLPIGVLKSSAQNLMNQEDVHSKQAPGTILKSQGMQVFVLEELDKTLFTLGFHKPAIVQHASSAKDLGPLLDGIWKTTTTKQAAKCLQKNLPSFLGVTSSEFRYLMNSQTRLPDNYLPLLPAIIDRFDNTLPLTGQAQIIFRRFLPLQGKEFQ</code>
            </div>
          </section>

          <section>
            <h2 className="text-2xl font-semibold text-white mb-4">FAQ</h2>
            <div className="space-y-4">
              <div>
                <h3 className="font-semibold text-white mb-2">What is the minimum sequence length?</h3>
                <p className="text-blue-200">Sequences should be at least 10 amino acids long.</p>
              </div>
              <div>
                <h3 className="font-semibold text-white mb-2">Are my sequences stored?</h3>
                <p className="text-blue-200">No, sequences are processed in real-time and not permanently stored.</p>
              </div>
              <div>
                <h3 className="font-semibold text-white mb-2">What models are used?</h3>
                <p className="text-blue-200">We use custom-trained MLP classifiers, ProteinLM models, and sequence generators.</p>
              </div>
            </div>
          </section>
        </div>
      </div>
    </div>
  );
}

