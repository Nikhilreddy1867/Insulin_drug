import { Link, useLocation } from 'react-router-dom';

interface NavbarProps {
  user: { id: string; username: string } | null;
  onLoginClick: () => void;
}

export default function Navbar({ user, onLoginClick }: NavbarProps) {
  const location = useLocation();

  return (
    <nav className="bg-gradient-to-br from-blue-900 via-indigo-900 to-purple-900 sticky top-0 z-50">
      {/* Main Navbar */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          {/* Logo */}
          <Link to="/" className="flex items-center gap-2">
            <div className="flex flex-col gap-1">
              <div className="flex gap-1">
                <div className="w-2 h-2 bg-white rounded"></div>
                <div className="w-2 h-2 bg-white rounded"></div>
              </div>
              <div className="flex gap-1">
                <div className="w-2 h-2 bg-white rounded"></div>
                <div className="w-2 h-2 bg-white rounded"></div>
              </div>
            </div>
            <span className="text-xl font-bold text-white">Insulin Drug Synthesis</span>
          </Link>

          {/* Navigation Links */}
          <div className="hidden md:flex items-center gap-6">
            <Link
              to="/models"
              className={`text-sm font-medium transition-colors ${
                location.pathname === '/models'
                  ? 'text-blue-300'
                  : 'text-white/90 hover:text-blue-300'
              }`}
            >
              Models
            </Link>
            <Link
              to="/about"
              className={`text-sm font-medium transition-colors ${
                location.pathname === '/about'
                  ? 'text-blue-300'
                  : 'text-white/90 hover:text-blue-300'
              }`}
            >
              About
            </Link>
            <Link
              to="/features"
              className={`text-sm font-medium transition-colors ${
                location.pathname === '/features'
                  ? 'text-blue-300'
                  : 'text-white/90 hover:text-blue-300'
              }`}
            >
              Features
            </Link>
            <Link
              to="/documentation"
              className={`text-sm font-medium transition-colors ${
                location.pathname === '/documentation'
                  ? 'text-blue-300'
                  : 'text-white/90 hover:text-blue-300'
              }`}
            >
              Documentation
            </Link>
          </div>

          {/* Right Side Actions */}
          <div className="flex items-center gap-4">
            {user ? (
              <Link
                to="/dashboard"
                className="px-4 py-2 bg-white/10 hover:bg-white/20 backdrop-blur-sm border border-white/20 text-white rounded-lg text-sm font-medium transition-colors"
              >
                Dashboard
              </Link>
            ) : (
              <>
                <button
                  onClick={onLoginClick}
                  className="px-4 py-2 bg-white/10 hover:bg-white/20 backdrop-blur-sm border border-white/20 text-white rounded-lg text-sm font-medium transition-colors hidden sm:block"
                >
                  Login
                </button>
                <button
                  onClick={onLoginClick}
                  className="px-4 py-2 bg-lime-400 hover:bg-lime-500 text-black rounded-lg text-sm font-semibold transition-colors"
                >
                  Sign up
                </button>
              </>
            )}
          </div>
        </div>
      </div>
    </nav>
  );
}

