import Home from '/src/pages/Home.jsx';
import Account from '/src/pages/Account.jsx';
import Login from '/src/components/Login/Login.jsx';
import Register from '/src/components/Register/Register.jsx';
import Profile from '/src/pages/Profile.jsx';
import Map from '/src/pages/map.jsx';
import {Routes, Route } from 'react-router-dom';

function App() {
  return (
      <Routes>
        <Route path='/' element={<Home />} />
        <Route path="/account" element={<Account />}>
          <Route path="login" element={<Login />} />
          <Route path="register" element={<Register />} />
        </Route>
        <Route path='/profile' element={<Profile/>}/>
        <Route path='/map' element={<Map/>}/>
      </Routes>
  );
}

export default App;
