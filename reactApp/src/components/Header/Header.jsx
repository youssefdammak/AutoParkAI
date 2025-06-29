import './Header.css'
import { Link } from 'react-router-dom';
function Header(){
    return (
        <header>
            <div className="header-container">
                <div className="logo">
                    <i className="fas fa-parking"></i>
                    <div>
                        <h1>AutoParkAI</h1>
                        <p>Smart Parking Management</p>
                    </div>
                </div>
                <nav>
                    <ul>
                        <li><Link to="/"><i className="fas fa-home"></i> Home</Link></li>
                        <li><Link to="/map"><i className="fas fa-map-marker-alt"></i> Parking Map</Link></li>
                        <li><Link to="/history"><i className="fas fa-history"></i> History</Link></li>
                        <li><Link to="/settings"><i className="fas fa-cog"></i> Settings</Link></li>
                        <li><Link to="/account/login"><i className="fas fa-user"></i> Account</Link></li>
                    </ul>
                </nav>
            </div>
        </header>
    );
}

export default Header;