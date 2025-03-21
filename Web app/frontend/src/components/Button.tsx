import React from 'react';

interface ButtonProps {
  text: string;
  onClick: () => void;
  variant?: 'primary' | 'secondary';
}

const Button: React.FC<ButtonProps> = ({ text, onClick, variant = 'primary' }) => {
  const buttonStyles = variant === 'primary' 
    ? 'bg-blue-600 text-white hover:bg-blue-700' 
    : 'bg-green-600 text-white hover:bg-green-700';

  return (
    <button
      onClick={onClick}
      className={`px-6 py-3 rounded-md ${buttonStyles} focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-opacity-50`}
    >
      {text}
    </button>
  );
};

export default Button;

