
import React, { useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { ToastContainer } from 'react-toastify';
import { handleError, handleSuccess } from '../utils';

function Signup() {

    const [signupInfo, setSignupInfo] = useState({
        username: '',
        email: '',
        password: ''
    })

    const navigate = useNavigate();
    const handleChange = (e) => {
        const { name, value } = e.target;
        // console.log(name, value);
        const copySignupInfo = { ...signupInfo };
        copySignupInfo[name] = value;
        setSignupInfo(copySignupInfo);
    }

    const handleSignup = async (e) => {
        e.preventDefault();
        const { username, email, password } = signupInfo;
        if (!username || !email || !password) {
            return handleError('username, email and password are required')
        }
        try {
            const response = await fetch("/api/auth/register", {
                method: "POST",
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(signupInfo)
            });

            const result = await response.json();

            if (response.ok) {
                // Success case - backend returns user object directly
                handleSuccess('User registered successfully!');
                setTimeout(() => {
                    navigate('/login')
                }, 1000)
            } else {
                // Error case - backend returns HTTPException with detail
                const errorMessage = result.detail || 'Registration failed';
                handleError(errorMessage);
            }

            console.log(result);
        } catch (err) {
            handleError('Network error or server unavailable');
            console.error('Registration error:', err);
        }
    }
    return (
        <div className='min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 flex items-center justify-center px-4 sm:px-6 lg:px-8'>
            <div className='max-w-md w-full space-y-8'>
                <div className='bg-white rounded-xl shadow-2xl p-8'>
                    <div className='text-center mb-8'>
                        <h1 className='text-3xl font-bold text-gray-900 mb-2'>Create Account</h1>

                    </div>

                    <form onSubmit={handleSignup} className='space-y-6'>
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
                                value={signupInfo.username}
                                className='w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 transition duration-200 ease-in-out placeholder-gray-400'
                            />
                        </div>

                        <div className='space-y-1'>
                            <label
                                htmlFor='email'
                                className='block text-sm font-medium text-gray-700 mb-2'
                            >
                                Email Address
                            </label>
                            <input
                                onChange={handleChange}
                                type='email'
                                name='email'
                                id='email'
                                placeholder='Enter your email...'
                                value={signupInfo.email}
                                className='w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 transition duration-200 ease-in-out placeholder-gray-400'
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
                                value={signupInfo.password}
                                className='w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 transition duration-200 ease-in-out placeholder-gray-400'
                            />
                        </div>

                        <button
                            type='submit'
                            className='w-full bg-indigo-600 hover:bg-indigo-700 text-white font-medium py-3 px-4 rounded-lg transition duration-200 ease-in-out transform hover:scale-[1.02] focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2 shadow-lg'
                        >
                            Create Account
                        </button>

                        <div className='text-center pt-4'>
                            <span className='text-gray-600'>
                                Already have an account?{' '}
                                <Link
                                    to="/login"
                                    className='text-indigo-600 hover:text-indigo-800 font-medium transition duration-200 ease-in-out hover:underline'
                                >
                                    Sign In
                                </Link>
                            </span>
                        </div>
                    </form>
                </div>
            </div>
            <ToastContainer />
        </div>
    )
}

export default Signup
