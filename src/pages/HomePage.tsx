import { useNavigate } from 'react-router-dom';
import { Activity, Brain, Dna, FlaskConical } from 'lucide-react';

export default function HomePage() {
  const navigate = useNavigate();

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-900 via-indigo-900 to-purple-900">
      {/* Hero Section */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-20">
        <div className="grid md:grid-cols-2 gap-12 items-center">
          {/* Left Side */}
          <div>
            <h1 className="text-5xl md:text-6xl font-bold text-blue-200 mb-4">
              Insulin Drug Synthesis
            </h1>
            <h2 className="text-4xl md:text-5xl font-bold text-white mb-6">
              Where researchers come to innovate.
            </h2>
          </div>

          {/* Right Side */}
          <div>
            <h3 className="text-2xl font-semibold text-white mb-4">
              The All-in-one Drug Discovery Platform.
            </h3>
            <p className="text-blue-200 mb-8 text-lg leading-relaxed">
              Analyze protein sequences, generate variants, convert to molecular structures—all from your browser. 
              Powered by advanced AI and machine learning models.
            </p>
            <button
              onClick={() => navigate('/dashboard')}
              className="px-8 py-4 bg-lime-400 hover:bg-lime-500 text-black font-semibold rounded-lg transition-colors text-lg"
            >
              Get Started - For Free!
            </button>
          </div>
        </div>
      </div>

      {/* Feature Showcase Grid */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-16">
        <div className="grid md:grid-cols-3 gap-6">
          {/* Link-in-bio / Analysis Card */}
          <div className="bg-white rounded-xl border border-gray-200 p-6 shadow-sm hover:shadow-md transition-shadow">
            <div className="mb-4">
              <h3 className="text-xl font-bold text-gray-900 mb-2">Sequence Analysis</h3>
              <p className="text-gray-600 text-sm">Classify protein sequences with AI-powered predictions</p>
            </div>
            <div className="bg-gray-100 rounded-lg p-4 mb-4 h-64 flex items-center justify-center">
              <div className="text-center">
                <div className="w-32 h-48 bg-white rounded-lg border-2 border-gray-300 mx-auto mb-4 shadow-inner flex items-center justify-center">
                  <Activity className="w-12 h-12 text-blue-500" />
                </div>
                <div className="space-y-2">
                  <div className="h-3 bg-gray-300 rounded w-24 mx-auto"></div>
                  <div className="h-2 bg-gray-200 rounded w-16 mx-auto"></div>
                </div>
              </div>
            </div>
            <div className="bg-blue-50 rounded-lg p-3 mb-2">
              <div className="text-xs text-gray-500 mb-1">Prediction Confidence</div>
              <div className="h-2 bg-blue-200 rounded-full">
                <div className="h-full bg-blue-500 rounded-full" style={{ width: '94%' }}></div>
              </div>
            </div>
          </div>

          {/* Media Kit / Models Card */}
          <div className="bg-white rounded-xl border border-gray-200 p-6 shadow-sm hover:shadow-md transition-shadow">
            <div className="mb-4">
              <h3 className="text-xl font-bold text-gray-900 mb-2">Model Library</h3>
              <p className="text-gray-600 text-sm">Access our collection of trained ML models</p>
            </div>
            <div className="bg-gray-100 rounded-lg p-4 mb-4 h-64 flex items-center justify-center">
              <div className="text-center">
                <div className="w-40 h-32 bg-white rounded-lg border-2 border-gray-300 mx-auto mb-4 shadow-inner flex items-center justify-center">
                  <Dna className="w-12 h-12 text-purple-500" />
                </div>
                <div className="space-y-2">
                  <div className="h-3 bg-gray-300 rounded w-28 mx-auto"></div>
                  <div className="h-2 bg-gray-200 rounded w-20 mx-auto"></div>
                </div>
              </div>
            </div>
            <div className="bg-purple-50 rounded-lg p-3">
              <div className="text-xs text-gray-500 mb-1">Model Performance</div>
              <div className="h-2 bg-purple-200 rounded-full">
                <div className="h-full bg-purple-500 rounded-full" style={{ width: '87%' }}></div>
              </div>
            </div>
          </div>

          {/* Email Marketing / Generation Card */}
          <div className="bg-white rounded-xl border border-gray-200 p-6 shadow-sm hover:shadow-md transition-shadow">
            <div className="mb-4">
              <h3 className="text-xl font-bold text-gray-900 mb-2">Sequence Generation</h3>
              <p className="text-gray-600 text-sm">Generate new protein variants with similarity metrics</p>
            </div>
            <div className="bg-gray-100 rounded-lg p-4 mb-4 h-64 flex items-center justify-center">
              <div className="text-center">
                <div className="w-32 h-40 bg-white rounded-lg border-2 border-gray-300 mx-auto mb-4 shadow-inner flex items-center justify-center">
                  <FlaskConical className="w-12 h-12 text-green-500" />
                </div>
                <div className="space-y-2">
                  <div className="h-3 bg-gray-300 rounded w-20 mx-auto"></div>
                  <div className="h-2 bg-gray-200 rounded w-14 mx-auto"></div>
                </div>
              </div>
            </div>
            <div className="bg-green-50 rounded-lg p-3">
              <div className="text-xs text-gray-500 mb-1">Generation Quality</div>
              <div className="h-2 bg-green-200 rounded-full">
                <div className="h-full bg-green-500 rounded-full" style={{ width: '91%' }}></div>
              </div>
            </div>
          </div>

          {/* Online Store / SMILES Card */}
          <div className="bg-white rounded-xl border border-gray-200 p-6 shadow-sm hover:shadow-md transition-shadow">
            <div className="mb-4">
              <h3 className="text-xl font-bold text-gray-900 mb-2">SMILES Conversion</h3>
              <p className="text-gray-600 text-sm">Convert protein sequences to molecular structures</p>
            </div>
            <div className="bg-gray-100 rounded-lg p-4 mb-4 h-64 flex items-center justify-center">
              <div className="text-center">
                <div className="w-40 h-32 bg-white rounded-lg border-2 border-gray-300 mx-auto mb-4 shadow-inner flex items-center justify-center">
                  <Activity className="w-12 h-12 text-teal-500" />
                </div>
                <div className="space-y-2">
                  <div className="h-3 bg-gray-300 rounded w-24 mx-auto"></div>
                  <div className="h-2 bg-gray-200 rounded w-18 mx-auto"></div>
                </div>
              </div>
            </div>
            <div className="bg-teal-50 rounded-lg p-3">
              <div className="text-xs text-gray-500 mb-1">Conversion Rate</div>
              <div className="h-2 bg-teal-200 rounded-full">
                <div className="h-full bg-teal-500 rounded-full" style={{ width: '89%' }}></div>
              </div>
            </div>
          </div>

          {/* AI Powered Card */}
          <div className="bg-gradient-to-br from-green-600 to-green-800 rounded-xl p-6 shadow-sm hover:shadow-md transition-shadow">
            <div className="flex flex-col items-center justify-center h-full min-h-[300px]">
              <div className="bg-green-500 rounded-full p-6 mb-4">
                <Brain className="w-16 h-16 text-white" />
              </div>
              <h3 className="text-2xl font-bold text-white">Powered by AI</h3>
            </div>
          </div>

          {/* Empty/Placeholder Card */}
          <div className="bg-white rounded-xl border border-gray-200 p-6 shadow-sm hover:shadow-md transition-shadow">
            <div className="h-64 flex items-center justify-center text-gray-400">
              <div className="text-center">
                <div className="w-24 h-24 border-2 border-dashed border-gray-300 rounded-lg mx-auto mb-4 flex items-center justify-center">
                  <Activity className="w-8 h-8 text-gray-300" />
                </div>
                <p className="text-sm">Coming Soon</p>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Creator/Brand Showcase Section */}
      <div className="bg-white/10 backdrop-blur-sm border-t border-white/20 py-16">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <h3 className="text-2xl font-bold text-white mb-8 text-center">
            Trusted by Researchers and Institutions
          </h3>
          <div className="flex gap-6 overflow-x-auto pb-4 scrollbar-hide">
            {[
              { name: 'BioLab Research', category: 'Biochemistry Lab', followers: '5.2k' },
              { name: 'Pharma Innovations', category: 'Pharmaceutical Company', followers: '24.8m' },
              { name: 'Dr. Elena Chen', category: 'Molecular Biologist', followers: '646k' },
              { name: 'Drug Discovery Hub', category: 'Research Institute', followers: '373k' },
              { name: 'Protein Analytics', category: 'Data Science Team', followers: '222.6k' },
              { name: 'Dr. James Mitchell', category: 'Bioinformatics', followers: '3.3m' },
              { name: 'Molecular Dynamics Lab', category: 'Research Group', followers: '22.4m' },
              { name: 'AI Drug Design', category: 'AI Research', followers: '1m' },
              { name: 'Biotech Solutions', category: 'Biotech Company', followers: '19.1m' },
            ].map((item, index) => (
              <div key={index} className="flex-shrink-0 w-48 text-center">
                <div className="w-24 h-24 bg-white/20 rounded-full mx-auto mb-3 flex items-center justify-center">
                  <span className="text-2xl font-bold text-white">
                    {item.name.charAt(0)}
                  </span>
                </div>
                <h4 className="font-semibold text-white mb-1">{item.name}</h4>
                <p className="text-xs text-blue-200 mb-1">{item.category}</p>
                <p className="text-xs text-blue-300">{item.followers} users</p>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

