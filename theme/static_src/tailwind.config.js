/**
 * This is a minimal config.
 *
 * If you need the full config, get it from here:
 * https://unpkg.com/browse/tailwindcss@latest/stubs/defaultConfig.stub.js
 */

module.exports = {
    content: [
        
        '../templates/**/*.html',
        "./verifier/templates/**/*.html",
        '../../templates/**/*.html',
        '../../**/templates/**/*.html',

       
    ],
    theme: {
        extend: {
          keyframes: {
            fadeOut: {
              '0%': { opacity: '1' }, // Fully visible
              '100%': { opacity: '0' } // Fully invisible
            },
            wiggle: {
              '0%': { transform: 'translate(-20px,0px)' },
              '50%': { transform: 'translate(50px,0px)'},
              '100%': { transform: 'translate(-20px,0px)'},
              
            },
            ziggle: {
              '0%': { transform: 'translate(20px,0px)' },
              '50%': { transform: 'translate(-50px,0px)'},
              '100%': { transform: 'translate(20px,0px)'},
            }
          },
          animation: {
            fadeOut: 'fadeOut 5s forwards',
            wiggle: 'wiggle 10s ease-in-out infinite',
            ziggle: 'ziggle 10s ease-in-out infinite', // Adjust duration as needed
          },
        },
      },
    plugins: [
        /**
         * '@tailwindcss/forms' is the forms plugin that provides a minimal styling
         * for forms. If you don't like it or have own styling for forms,
         * comment the line below to disable '@tailwindcss/forms'.
         */
        require('@tailwindcss/forms'),
        require('@tailwindcss/typography'),
        require('@tailwindcss/aspect-ratio'),
    ],
}
