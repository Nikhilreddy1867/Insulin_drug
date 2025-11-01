export default function AboutPage() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-900 via-indigo-900 to-purple-900 py-16">
      <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
        <h1 className="text-4xl font-bold text-white mb-8">About Insulin Drug Synthesis</h1>
        
        <div className="bg-white/10 backdrop-blur-sm rounded-xl border border-white/20 p-8 shadow-sm space-y-6">
          <section>
            <h2 className="text-2xl font-semibold text-white mb-4">Our Mission</h2>
            <p className="text-blue-200 leading-relaxed">
              Insulin Drug Synthesis is an advanced AI-powered platform designed to revolutionize 
              the process of drug discovery and protein analysis. We combine cutting-edge machine 
              learning models with intuitive user interfaces to make complex biochemical analysis 
              accessible to researchers worldwide.
            </p>
          </section>

          <section>
            <h2 className="text-2xl font-semibold text-white mb-4">What We Do</h2>
            <ul className="space-y-3 text-blue-200">
              <li className="flex items-start">
                <span className="text-blue-500 mr-2">•</span>
                <span><strong>Protein Sequence Classification:</strong> Analyze and classify protein sequences 
                using advanced ML models to predict pathogenicity and functionality.</span>
              </li>
              <li className="flex items-start">
                <span className="text-purple-500 mr-2">•</span>
                <span><strong>Sequence Generation:</strong> Generate new protein sequence variants with 
                similarity metrics and ranking algorithms.</span>
              </li>
              <li className="flex items-start">
                <span className="text-green-500 mr-2">•</span>
                <span><strong>SMILES Conversion:</strong> Convert protein sequences to SMILES molecular 
                structures for drug design applications.</span>
              </li>
            </ul>
          </section>

          <section>
            <h2 className="text-2xl font-semibold text-white mb-4">Technology Stack</h2>
            <p className="text-blue-200 leading-relaxed">
              Our platform leverages state-of-the-art deep learning architectures including custom 
              ProteinLM models, MLP classifiers, and PCA dimensionality reduction techniques. 
              The backend is built with Python and Flask, while the frontend utilizes React and 
              TypeScript for a seamless user experience.
            </p>
          </section>

          <section>
            <h2 className="text-2xl font-semibold text-white mb-4">Contact</h2>
            <p className="text-blue-200">
              For inquiries, support, or collaboration opportunities, please reach out through 
              our platform dashboard or contact our research team.
            </p>
          </section>
        </div>
      </div>
    </div>
  );
}

