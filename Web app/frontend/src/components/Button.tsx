// frontend/src/components/Button.tsx

import React from 'react';

interface ButtonProps {
  text: string;
  onClick: () => void;
  type?: 'button' | 'submit' | 'reset';
  className?: string; // Allow custom class names for additional styling
}

const Button: React.FC<ButtonProps> = ({ text, onClick, type = 'button', className = '' }) => {
  return (
    <button
      type={type}
      onClick={onClick}
      className={`px-4 py-2 bg-blue-500 text-white rounded-md hover:bg-blue-600 transition-all duration-300 ${className}`}
    >
      {text}
    </button>
  );
};

export default Button;
