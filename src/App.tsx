import { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { Loader } from 'lucide-react';
import LoginPage from './LoginPage';
import Navbar from './components/Navbar';
import HomePage from './pages/HomePage';
import AboutPage from './pages/AboutPage';
import ModelsPage from './pages/ModelsPage';
import FeaturesPage from './pages/FeaturesPage';
import DocumentationPage from './pages/DocumentationPage';
import DashboardPage from './pages/DashboardPage';
import AlphaFold2Page from './pages/AlphaFold2Page';
import DockingPage from './pages/DockingPage';

interface User {
  id: string;
  username: string;
}

function App() {
  const [user, setUser] = useState<User | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isDarkMode, setIsDarkMode] = useState(false);
  const [showLoginPage, setShowLoginPage] = useState(false);

  // Check authentication status on component mount
  useEffect(() => {
    checkAuthStatus();
    checkGoogleAuthCallback();
  }, []);

  const checkAuthStatus = async () => {
    try {
      const response = await fetch('http://localhost:5001/api/check-auth', {
        credentials: 'include',
      });
      
      if (response.ok) {
        const data = await response.json();
        if (data.authenticated && data.user) {
          setUser(data.user);
        }
      }
    } catch (err) {
      console.error('Auth check failed:', err);
    } finally {
      setIsLoading(false);
    }
  };

  const checkGoogleAuthCallback = () => {
    const urlParams = new URLSearchParams(window.location.search);
    const authStatus = urlParams.get('auth');
    const username = urlParams.get('user');
    
    if (authStatus === 'success' && username) {
      setUser({ id: username, username: username });
      window.history.replaceState({}, document.title, window.location.pathname);
    }
  };

  const handleLogin = (userData: User) => {
    setUser(userData);
    setShowLoginPage(false);
  };

  const handleLogout = async () => {
    try {
      await fetch('http://localhost:5001/api/logout', {
        method: 'POST',
        credentials: 'include',
      });
      setUser(null);
    } catch (err) {
      console.error('Logout error:', err);
    }
  };

  const toggleTheme = () => {
    setIsDarkMode(!isDarkMode);
  };

  // Show loading screen while checking authentication
  if (isLoading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-blue-900 via-indigo-900 to-purple-900 flex items-center justify-center animate-fadeIn">
        <div className="text-center">
          <div className="w-16 h-16 border-4 border-white border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
          <p className="text-white text-lg">Loading...</p>
        </div>
      </div>
    );
  }

  // Show login page if user clicks login button
  if (showLoginPage && !user) {
    return (
      <div className="animate-fadeIn">
        <LoginPage onLogin={handleLogin} />
      </div>
    );
  }
  
  return (
    <Router>
      <div className={`min-h-screen transition-all duration-500 ease-in-out bg-gradient-to-br from-blue-900 via-indigo-900 to-purple-900`}>
        <Navbar user={user} onLoginClick={() => setShowLoginPage(true)} />
        
        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route path="/about" element={<AboutPage />} />
          <Route path="/models" element={<ModelsPage />} />
          <Route path="/features" element={<FeaturesPage />} />
          <Route path="/documentation" element={<DocumentationPage />} />
          <Route path="/alphafold2" element={<AlphaFold2Page />} />
          <Route path="/docking" element={<DockingPage />} />
          <Route
            path="/dashboard"
            element={
              <DashboardPage
                user={user}
                onLogout={handleLogout}
                isDarkMode={isDarkMode}
                toggleTheme={toggleTheme}
              />
            }
          />
        </Routes>
      </div>
    </Router>
  );
}

export default App;
