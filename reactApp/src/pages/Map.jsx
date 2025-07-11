import Header from '../components/Header/Header.jsx'
import ParkingMap from '../components/ParkingMap/ParkingMap.jsx'
import Footer from '../components/Footer/Footer.jsx'

function Map() {
  return(
    <div className='app'>
        <Header></Header>
        <div className="main-container">
            <ParkingMap></ParkingMap>
        </div>
        <Footer></Footer>
    </div>
  );
}

export default Map
