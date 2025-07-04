import Header from '../components/Header/Header.jsx'
import ProfileDashboard from '../components/ProfileDashboard/ProfileDashboard.jsx'

function Profile() {
  return(
    <div className='app'>
        <Header></Header>
        <div className="main-container">
            <ProfileDashboard></ProfileDashboard>
        </div>
    </div>
  );
}

export default Profile
