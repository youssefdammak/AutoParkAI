import './Footer.css';
import { Link } from 'react-router-dom';

function Footer() {
  return (
    <footer className="site-footer">
      <div className="footer-container">
        <div className="footer-branding">
          <i className="fas fa-parking"></i>
          <div>
            <h2>AutoParkAI</h2>
            <p>Smart Parking Management</p>
          </div>
        </div>
        <div className="footer-links">
          <Link to="/">Home</Link>
          <Link to="/map">Map</Link>
          <Link to="/account/login">Login</Link>
          <Link to="/account/register">Register</Link>
        </div>
        <div className="footer-credits">
          <p>&copy; 2025 AutoParkAI. All rights reserved.</p>
        </div>
      </div>
    </footer>
  );
}

export default Footer;
