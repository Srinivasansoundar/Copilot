import React, { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom';
import { handleError, handleSuccess } from '../utils';
import { ToastContainer } from 'react-toastify';

function Home() {
    const [loggedInUser, setLoggedInUser] = useState('');
    const [products, setProducts] = useState('');
    const navigate = useNavigate();
    useEffect(() => {
        setLoggedInUser(localStorage.getItem('loggedInUser'))
    }, [])

    const handleLogout = async (e) => {
        try {
            const token = localStorage.getItem('token');

            if (token) {
                // Call backend logout endpoint to clear server-side session
                const response = await fetch("/api/auth/logout", {
                    method: "POST",
                    headers: {
                        'Authorization': `Bearer ${token}`,
                        'Content-Type': 'application/json'
                    }
                });

                if (response.ok) {
                    const result = await response.json();
                    handleSuccess(result.message || 'User Logged out');
                } else {
                    // Even if backend logout fails, still clear local storage
                    console.warn('Backend logout failed, clearing local storage anyway');
                    handleSuccess('User Logged out');
                }
            } else {
                // No token found, just show success message
                handleSuccess('User Logged out');
            }

            // Always clear local storage regardless of backend response
            localStorage.removeItem('token');
            localStorage.removeItem('loggedInUser');

            setTimeout(() => {
                navigate('/login');
            }, 1000);

        } catch (err) {
            // Even if there's a network error, still clear local storage
            console.error('Logout error:', err);
            localStorage.removeItem('token');
            localStorage.removeItem('loggedInUser');
            handleSuccess('User Logged out');
            setTimeout(() => {
                navigate('/login');
            }, 1000);
        }
    }



    return (
        <div className='m-4'>
            <h1>Welcome {loggedInUser}</h1>
            <button onClick={handleLogout} className='bg-red-500 p-2 w-[100px] font-bold text-center text-md rounded-md'>Logout</button>
            <ToastContainer />
        </div>
    )
}

export default Home