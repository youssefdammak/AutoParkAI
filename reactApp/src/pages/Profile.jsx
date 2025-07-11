import Header from '../components/Header/Header.jsx'
import ProfileDashboard from '../components/ProfileDashboard/ProfileDashboard.jsx'
import Footer from '../components/Footer/Footer.jsx'

function Profile() {
  return(
    <div className='app'>
        <Header></Header>
        <div className="main-container">
            <ProfileDashboard></ProfileDashboard>
        </div>
        <Footer></Footer>
    </div>
  );
}

export default Profile
