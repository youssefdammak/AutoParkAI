import './ParkingMap.css'; // External CSS for clarity

const rows = ['A', 'B'];
const spotsPerRow = 10;

// Example: You can replace this with fetched data
const occupiedSpots = ['A2', 'A4', 'A7', 'A10', 'B1', 'B3', 'B6', 'B9'];

function ParkingMap() {
  const isOccupied = (spot) => occupiedSpots.includes(spot);

  return (
    <div className='parkingMap'>
        <div className="card">
        <div className="card-title">Smart Parking Lot Map</div>

        <div className="road-label">Entrance ➡️</div>

        {rows.map((rowLabel) => (
            <div className="parking-lot" key={rowLabel}>
            <div className="row-label">{rowLabel}</div>
            {[...Array(spotsPerRow)].map((_, i) => {
                const spotId = `${rowLabel}${i + 1}`;
                return (
                <div
                    key={spotId}
                    className={`spot ${isOccupied(spotId) ? 'occupied' : 'available'}`}
                >
                    {spotId}
                </div>
                );
            })}
            <div className="row-label">{rowLabel}</div>
            </div>
        ))}

        <div className="road-label">⬅️ Exit</div>

        <div className="legend">
            <div className="legend-item">
            <div className="legend-box available-box"></div> Available
            </div>
            <div className="legend-item">
            <div className="legend-box occupied-box"></div> Occupied
            </div>
            <div className="legend-item">
            <div className="legend-box road-box"></div> Road
            </div>
        </div>
        </div>
    </div>
  );
}

export default ParkingMap;
