import Home from '/src/pages/Home.jsx';
import Account from '/src/pages/Account.jsx';
import Login from '/src/components/Login/Login.jsx';
import Register from '/src/components/Register/Register.jsx';
import { BrowserRouter, Routes, Route } from 'react-router-dom';

function App() {
  return (
      <Routes>
        <Route path='/' element={<Home />} />
        <Route path="/account" element={<Account />}>
          <Route path="login" element={<Login />} />
          <Route path="register" element={<Register />} />
        </Route>
      </Routes>
  );
}

export default App;
