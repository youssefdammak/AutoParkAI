import './Dashboard.css'

function Dashboard(){
    return(
        <section className="dashboard">
            <div className="card">
                <div className="card-header">
                    <h3 className="card-title">Parking Availability</h3>
                    <div className="card-icon">
                        <i className="fas fa-parking"></i>
                    </div>
                </div>
                <div className="stats">
                    <div className="stat-item">
                        <div className="stat-value available" id="available-spots">42</div>
                        <div className="stat-label">Available</div>
                    </div>
                    <div className="stat-item">
                        <div className="stat-value occupied" id="occupied-spots">58</div>
                        <div className="stat-label">Occupied</div>
                    </div>
                    <div className="stat-item">
                        <div className="stat-value" id="total-spots">100</div>
                        <div className="stat-label">Total</div>
                    </div>
                </div>
            </div>
            <div className="card">
                <div className="card-header">
                    <h3 className="card-title">Today's Activity</h3>
                    <div className="card-icon">
                        <i className="fas fa-chart-line"></i>
                    </div>
                </div>
                <div className="stats">
                    <div className="stat-item">
                        <div className="stat-value">124</div>
                        <div className="stat-label">Entries</div>
                    </div>
                    <div className="stat-item">
                        <div className="stat-value">98</div>
                        <div className="stat-label">Exits</div>
                    </div>
                    <div className="stat-item">
                        <div className="stat-value">86%</div>
                        <div className="stat-label">Capacity</div>
                    </div>
                </div>
            </div>
            
            <div className="card">
                <div className="card-header">
                    <h3 className="card-title">Average Stay</h3>
                    <div className="card-icon">
                        <i className="fas fa-clock"></i>
                    </div>
                </div>
                <div className="stats">
                    <div className="stat-item">
                        <div className="stat-value">2h 15m</div>
                        <div className="stat-label">Today</div>
                    </div>
                    <div className="stat-item">
                        <div className="stat-value">1h 52m</div>
                        <div className="stat-label">Yesterday</div>
                    </div>
                    <div className="stat-item">
                        <div className="stat-value">2h 04m</div>
                        <div className="stat-label">Weekly Avg</div>
                    </div>
                </div>
            </div>
        </section>
    );
}

export default Dashboard;