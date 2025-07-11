import Header from '../components/Header/Header.jsx'
import Footer from '../components/Footer/Footer.jsx'
import { Outlet } from 'react-router-dom';
function Account() {
  return(
    <div className='app'>
        <Header></Header>
        <div className="main-container">
            <Outlet></Outlet>
        </div>
        <Footer></Footer>
    </div>
  );
}

export default Account
