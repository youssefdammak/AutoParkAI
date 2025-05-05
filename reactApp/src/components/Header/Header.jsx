import './Header.css'
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
                        <li><a href="#"><i className="fas fa-home"></i> Home</a></li>
                        <li><a href="#"><i className="fas fa-map-marker-alt"></i> Parking Map</a></li>
                        <li><a href="#"><i className="fas fa-history"></i> History</a></li>
                        <li><a href="#"><i className="fas fa-cog"></i> Settings</a></li>
                        <li><a href="#"><i className="fas fa-user"></i> Account</a></li>
                    </ul>
                </nav>
            </div>
        </header>
    );
}

export default Header;