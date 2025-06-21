import React, { useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { ToastContainer } from 'react-toastify';
import { handleError, handleSuccess } from '../utils';

function Login() {

    const [loginInfo, setLoginInfo] = useState({
        username: '',
        password: ''
    })

    const navigate = useNavigate();

    const handleChange = (e) => {
        const { name, value } = e.target;
        console.log(name, value);
        const copyLoginInfo = { ...loginInfo };
        copyLoginInfo[name] = value;
        setLoginInfo(copyLoginInfo);
    }

    const handleLogin = async (e) => {
        e.preventDefault();
        // console.log("hello")
        const { username, password } = loginInfo;
        
        if (!username || !password) {
            return handleError('username and password are required')
        }
        try {
            // FastAPI OAuth2PasswordRequestForm expects form data, not JSON
            const formData = new FormData();
            formData.append('username', username);
            formData.append('password', password);
            console.log("hello")
            const response = await fetch("/api/auth/login", {
                method: "POST",
                body: formData // Send as form data, not JSON
            });
            // console.log('hello')
            const result = await response.json();

            if (response.ok) {
                // Success case - backend returns Token object with access_token and token_type
                const { access_token, token_type } = result;

                handleSuccess('Login successful!');
                localStorage.setItem('token', access_token);
                localStorage.setItem('loggedInUser', username); // Use username from loginInfo
                setTimeout(() => {
                    navigate('/chat')
                }, 1000)
            } else {
                // Error case - backend returns HTTPException with detail
                const errorMessage = result.detail || 'Login failed';
                handleError(errorMessage);
            }

            console.log(result);
        } catch (err) {
            handleError('Network error or server unavailable');
            console.error('Login error:', err);
        }
    }

    return (
        <div className='min-h-screen bg-gradient-to-br from-purple-50 to-blue-100 flex items-center justify-center px-4 sm:px-6 lg:px-8'>
            <div className='max-w-md w-full space-y-8'>
                <div className='bg-white rounded-xl shadow-2xl p-8'>
                    <div className='text-center mb-8'>
                        <h1 className='text-3xl font-bold text-gray-900 mb-2'>Welcome Back</h1>
                        <p className='text-gray-600'>Sign in to your account</p>
                    </div>

                    <div className='space-y-6'>
                        <div className='space-y-1'>
                            <label
                                htmlFor='username'
                                className='block text-sm font-medium text-gray-700 mb-2'
                            >
                                Username
                            </label>
                            <input
                                onChange={handleChange}
                                type='text'
                                name='username'
                                id='username'
                                autoFocus
                                placeholder='Enter your username...'
                                value={loginInfo.username}
                                className='w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition duration-200 ease-in-out placeholder-gray-400'
                            />
                        </div>

                        <div className='space-y-1'>
                            <label
                                htmlFor='password'
                                className='block text-sm font-medium text-gray-700 mb-2'
                            >
                                Password
                            </label>
                            <input
                                onChange={handleChange}
                                type='password'
                                name='password'
                                id='password'
                                placeholder='Enter your password...'
                                value={loginInfo.password}
                                className='w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition duration-200 ease-in-out placeholder-gray-400'
                            />
                        </div>

                        <button
                            onClick={handleLogin}
                            className='w-full bg-blue-600 hover:bg-blue-700 text-white font-medium py-3 px-4 rounded-lg transition duration-200 ease-in-out transform hover:scale-[1.02] focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 shadow-lg'
                        >
                            Sign In
                        </button>

                        <div className='text-center pt-4'>
                            <span className='text-gray-600'>
                                Don't have an account?{' '}
                                <a
                                    href="/signup"
                                    className='text-blue-600 hover:text-blue-800 font-medium transition duration-200 ease-in-out hover:underline'
                                >
                                    Sign Up
                                </a>
                            </span>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
    )
}

export default Login
