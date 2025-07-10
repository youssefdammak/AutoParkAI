import Header from '../components/Header/Header.jsx'
import ParkingMap from '../components/ParkingMap/ParkingMap.jsx'

function Map() {
  return(
    <div className='app'>
        <Header></Header>
        <div className="main-container">
            <ParkingMap></ParkingMap>
        </div>
    </div>
  );
}

export default Map
