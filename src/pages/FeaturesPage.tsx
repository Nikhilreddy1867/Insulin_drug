import { Activity, Dna, FlaskConical, Brain, Zap, Shield } from 'lucide-react';

export default function FeaturesPage() {
  const features = [
    {
      icon: Activity,
      title: 'Real-time Analysis',
      description: 'Get instant predictions and classifications for your protein sequences',
      color: 'blue',
    },
    {
      icon: Dna,
      title: 'Sequence Generation',
      description: 'Generate protein variants with advanced similarity metrics',
      color: 'purple',
    },
    {
      icon: FlaskConical,
      title: 'SMILES Conversion',
      description: 'Convert sequences to molecular structures for drug design',
      color: 'green',
    },
    {
      icon: Brain,
      title: 'AI-Powered',
      description: 'Leverage state-of-the-art machine learning models',
      color: 'indigo',
    },
    {
      icon: Zap,
      title: 'Fast Processing',
      description: 'Optimized algorithms for quick results',
      color: 'yellow',
    },
    {
      icon: Shield,
      title: 'Secure & Private',
      description: 'Your data is encrypted and kept confidential',
      color: 'red',
    },
  ];

  const colorClasses = {
    blue: 'bg-blue-50 text-blue-600 border-blue-200',
    purple: 'bg-purple-50 text-purple-600 border-purple-200',
    green: 'bg-green-50 text-green-600 border-green-200',
    indigo: 'bg-indigo-50 text-indigo-600 border-indigo-200',
    yellow: 'bg-yellow-50 text-yellow-600 border-yellow-200',
    red: 'bg-red-50 text-red-600 border-red-200',
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-900 via-indigo-900 to-purple-900 py-16">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <h1 className="text-4xl font-bold text-white mb-4">Features</h1>
        <p className="text-blue-200 mb-12 text-lg">
          Discover what makes our platform powerful for drug discovery and protein analysis
        </p>

        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6 mb-12">
          {features.map((feature, index) => {
            const Icon = feature.icon;
            return (
              <div
                key={index}
                className="bg-white/10 backdrop-blur-sm rounded-xl border border-white/20 p-6 shadow-sm hover:shadow-md transition-shadow"
              >
                <div className={`w-12 h-12 rounded-lg flex items-center justify-center mb-4 ${colorClasses[feature.color as keyof typeof colorClasses]}`}>
                  <Icon className="w-6 h-6" />
                </div>
                <h3 className="text-xl font-bold text-white mb-2">{feature.title}</h3>
                <p className="text-blue-200">{feature.description}</p>
              </div>
            );
          })}
        </div>

        <div className="bg-white/10 backdrop-blur-sm rounded-xl border border-white/20 p-8 shadow-sm">
          <h2 className="text-2xl font-bold text-white mb-6">How It Works</h2>
          <div className="space-y-6">
            <div className="flex gap-4">
              <div className="flex-shrink-0 w-8 h-8 bg-blue-600 text-white rounded-full flex items-center justify-center font-bold">
                1
              </div>
              <div>
                <h3 className="font-semibold text-white mb-2">Input Your Sequence</h3>
                <p className="text-blue-200">Enter your protein sequence in the dashboard</p>
              </div>
            </div>
            <div className="flex gap-4">
              <div className="flex-shrink-0 w-8 h-8 bg-purple-600 text-white rounded-full flex items-center justify-center font-bold">
                2
              </div>
              <div>
                <h3 className="font-semibold text-white mb-2">AI Analysis</h3>
                <p className="text-blue-200">Our models process and analyze your sequence</p>
              </div>
            </div>
            <div className="flex gap-4">
              <div className="flex-shrink-0 w-8 h-8 bg-green-600 text-white rounded-full flex items-center justify-center font-bold">
                3
              </div>
              <div>
                <h3 className="font-semibold text-white mb-2">Get Results</h3>
                <p className="text-blue-200">Receive detailed predictions and insights</p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

