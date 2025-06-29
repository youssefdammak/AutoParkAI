import Header from '../components/Header/Header.jsx'
import { Outlet } from 'react-router-dom';
function Account() {
  return(
    <div className='app'>
        <Header></Header>
        <div className="main-container">
            <Outlet></Outlet>
        </div>
    </div>
  );
}

export default Account
