import { Dna, Brain, Activity } from 'lucide-react';

export default function ModelsPage() {
  const models = [
    {
      name: 'Protein Classification MLP',
      description: 'Multi-layer perceptron for classifying protein sequences into pathogenicity categories',
      icon: Brain,
      color: 'blue',
      features: ['Pathogenicity prediction', '5-class classification', 'PCA dimensionality reduction'],
    },
    {
      name: 'Sequence Generator',
      description: 'Generates new protein sequence variants with similarity metrics',
      icon: Dna,
      color: 'purple',
      features: ['Variant generation', 'Levenshtein distance', 'Hamming distance calculation'],
    },
    {
      name: 'Protein to SMILES',
      description: 'Converts protein sequences to SMILES molecular structures',
      icon: Activity,
      color: 'green',
      features: ['Molecular structure conversion', 'Drug design support', 'Chemical representation'],
    },
    {
      name: 'Custom ProteinLM',
      description: 'Custom language model trained for protein sequence understanding',
      icon: Brain,
      color: 'indigo',
      features: ['Sequence embedding', 'Feature extraction', 'Transfer learning'],
    },
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-900 via-indigo-900 to-purple-900 py-16">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <h1 className="text-4xl font-bold text-white mb-4">Available Models</h1>
        <p className="text-blue-200 mb-12 text-lg">
          Explore our collection of trained machine learning models for protein analysis
        </p>

        <div className="grid md:grid-cols-2 gap-6">
          {models.map((model, index) => {
            const Icon = model.icon;
            const colorClasses = {
              blue: 'bg-blue-50 border-blue-200 text-blue-600',
              purple: 'bg-purple-50 border-purple-200 text-purple-600',
              green: 'bg-green-50 border-green-200 text-green-600',
              indigo: 'bg-indigo-50 border-indigo-200 text-indigo-600',
            };

            return (
              <div
                key={index}
                className="bg-white/10 backdrop-blur-sm rounded-xl border border-white/20 p-6 shadow-sm hover:shadow-md transition-shadow"
              >
                <div className="flex items-start gap-4 mb-4">
                  <div className={`p-3 rounded-lg ${colorClasses[model.color as keyof typeof colorClasses]}`}>
                    <Icon className="w-6 h-6" />
                  </div>
                  <div className="flex-1">
                    <h3 className="text-xl font-bold text-white mb-2">{model.name}</h3>
                    <p className="text-blue-200">{model.description}</p>
                  </div>
                </div>
                <div className="mt-4 pt-4 border-t border-white/20">
                  <h4 className="text-sm font-semibold text-white mb-2">Key Features:</h4>
                  <ul className="space-y-1">
                    {model.features.map((feature, idx) => (
                      <li key={idx} className="text-sm text-blue-200 flex items-start">
                        <span className="text-gray-400 mr-2">•</span>
                        <span>{feature}</span>
                      </li>
                    ))}
                  </ul>
                </div>
                <div className="mt-6">
                  <div className="bg-gray-100 rounded-lg p-4 h-32 flex items-center justify-center">
                    <span className="text-gray-400 text-sm">Model Architecture Preview</span>
                  </div>
                </div>
              </div>
            );
          })}
        </div>

        <div className="mt-12 bg-white/10 backdrop-blur-sm rounded-xl border border-white/20 p-8 shadow-sm">
          <h2 className="text-2xl font-bold text-white mb-4">Model Performance</h2>
          <p className="text-blue-200 mb-6">
            Our models are continuously trained and optimized on large datasets of protein sequences 
            to ensure high accuracy and reliability.
          </p>
          <div className="grid md:grid-cols-3 gap-6">
            <div className="text-center p-4 bg-white/10 backdrop-blur-sm rounded-lg">
              <div className="text-3xl font-bold text-blue-300 mb-2">94.2%</div>
              <div className="text-sm text-blue-200">Classification Accuracy</div>
            </div>
            <div className="text-center p-4 bg-white/10 backdrop-blur-sm rounded-lg">
              <div className="text-3xl font-bold text-purple-300 mb-2">89.7%</div>
              <div className="text-sm text-blue-200">Sequence Similarity</div>
            </div>
            <div className="text-center p-4 bg-white/10 backdrop-blur-sm rounded-lg">
              <div className="text-3xl font-bold text-green-300 mb-2">91.3%</div>
              <div className="text-sm text-blue-200">SMILES Conversion</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

