import { useEffect, useState } from 'react';
import './Dashboard.css';

function Dashboard() {
    const total = 20; // total parking spots

    const [available, setAvailable] = useState(total);
    const [occupied, setOccupied] = useState(0);

    // Function to calculate availability based on plates in localStorage
    const calculateSpots = () => {
        const plates = JSON.parse(localStorage.getItem('plates') || '[]');

        const occupiedPlates = new Set();

        plates.forEach(entry => {
            if (entry.status.toLowerCase() === 'entry') {
                occupiedPlates.add(entry.plate);
            } else if (entry.status.toLowerCase() === 'exit') {
                occupiedPlates.delete(entry.plate);
            }
        });

        const occ = occupiedPlates.size;
        setOccupied(occ);
        setAvailable(total - occ);
    };

    useEffect(() => {
        calculateSpots(); // initial calculation on mount

        // Listener for storage changes in other tabs/windows
        const handleStorage = (event) => {
            if (event.key === 'plates') {
                calculateSpots();
            }
        };

        // Listener for custom event in the same tab
        const handleCustomEvent = () => {
            calculateSpots();
        };

        window.addEventListener('storage', handleStorage);
        window.addEventListener('platesUpdated', handleCustomEvent);

        // Cleanup listeners on unmount
        return () => {
            window.removeEventListener('storage', handleStorage);
            window.removeEventListener('platesUpdated', handleCustomEvent);
        };
    }, []);

    return (
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
                        <div className="stat-value available" id="available-spots">{available}</div>
                        <div className="stat-label">Available</div>
                    </div>
                    <div className="stat-item">
                        <div className="stat-value occupied" id="occupied-spots">{occupied}</div>
                        <div className="stat-label">Occupied</div>
                    </div>
                    <div className="stat-item">
                        <div className="stat-value" id="total-spots">{total}</div>
                        <div className="stat-label">Total</div>
                    </div>
                </div>
            </div>

            {/* You can keep your other cards as is */}
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
