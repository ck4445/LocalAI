// animations.js - Advanced loading animations for AI chat
const AnimationManager = (function() {
    let currentAnimation = localStorage.getItem('loaderAnimation') || 'simple';
    let gracePointInterval = null;
    
    // Inject Elden Ring animation styles
    function injectEldenRingStyles() {
        if (document.getElementById('elden-ring-styles')) return;
        
        const styles = document.createElement('style');
        styles.id = 'elden-ring-styles';
        styles.textContent = `
            /* Elden Ring Grace Loader */
            .grace-loader {
                display: inline-block;
                width: 1.2em;
                height: 1.2em;
                position: relative;
                vertical-align: middle;
                margin-right: -0.2em;
            }

            .grace-loader.fading {
                animation: fadeToText 0.6s ease-out forwards;
            }

            @keyframes fadeToText {
                0% {
                    opacity: 1;
                    transform: scale(1);
                    filter: blur(0px);
                }
                60% {
                    opacity: 0.5;
                    transform: scale(1.2);
                    filter: blur(2px);
                }
                100% {
                    opacity: 0;
                    transform: scale(0.8);
                    filter: blur(4px);
                }
            }

            /* Outer Golden Ring */
            .golden-ring {
                position: absolute;
                width: 100%;
                height: 100%;
                animation: rotateRing 6s linear infinite;
            }

            @keyframes rotateRing {
                from { transform: rotate(0deg); }
                to { transform: rotate(360deg); }
            }

            .ring-path {
                fill: none;
                stroke: url(#goldGradient);
                stroke-width: 1.5;
                opacity: 0.9;
                filter: url(#glow);
                animation: pulseRing 3s ease-in-out infinite;
            }

            @keyframes pulseRing {
                0%, 100% {
                    stroke-width: 1.5;
                    opacity: 0.9;
                }
                50% {
                    stroke-width: 2;
                    opacity: 1;
                }
            }

            /* Runic Glyphs */
            .rune-glyph {
                fill: #d4af37;
                opacity: 0.3;
                animation: flickerGlyph var(--flicker-duration) ease-in-out infinite;
                transform-origin: center;
                filter: url(#glow);
            }

            @keyframes flickerGlyph {
                0%, 100% { 
                    opacity: 0.2;
                    transform: scale(1);
                }
                25% { 
                    opacity: 0.6;
                    transform: scale(1.05);
                }
                50% { 
                    opacity: 0.3;
                    transform: scale(1);
                }
                75% { 
                    opacity: 0.8;
                    transform: scale(1.02);
                }
            }

            /* Breathing Aura */
            .golden-aura {
                position: absolute;
                width: 150%;
                height: 150%;
                top: -25%;
                left: -25%;
                background: radial-gradient(circle at center, 
                    rgba(212, 175, 55, 0.4) 0%, 
                    rgba(255, 215, 0, 0.2) 20%,
                    rgba(212, 175, 55, 0.1) 40%,
                    transparent 70%);
                animation: breatheAura 3s ease-in-out infinite;
                pointer-events: none;
            }

            @keyframes breatheAura {
                0%, 100% {
                    transform: scale(0.8);
                    opacity: 0.6;
                }
                50% {
                    transform: scale(1.1);
                    opacity: 1;
                }
            }

            /* Center Grace Point */
            .grace-point {
                position: absolute;
                width: 4px;
                height: 4px;
                background: radial-gradient(circle, #fff 0%, #ffd700 50%, transparent 100%);
                border-radius: 50%;
                top: 50%;
                left: 50%;
                transform: translate(-50%, -50%);
                box-shadow: 0 0 10px #ffd700, 0 0 20px #d4af37;
                animation: glowPoint 1.5s ease-in-out infinite;
            }

            @keyframes glowPoint {
                0%, 100% {
                    box-shadow: 0 0 10px #ffd700, 0 0 20px #d4af37;
                    opacity: 0.8;
                }
                50% {
                    box-shadow: 0 0 20px #ffd700, 0 0 40px #d4af37, 0 0 60px rgba(212, 175, 55, 0.5);
                    opacity: 1;
                }
            }

            /* Light theme adjustments */
            [data-theme="light"] .golden-aura {
                background: radial-gradient(circle at center, 
                    rgba(212, 175, 55, 0.25) 0%, 
                    rgba(255, 215, 0, 0.15) 20%,
                    rgba(212, 175, 55, 0.08) 40%,
                    transparent 70%);
            }

            [data-theme="light"] .rune-glyph {
                fill: #b8860b;
            }
        `;
        document.head.appendChild(styles);
    }

    // Create Elden Ring loader element
    function createEldenRingLoader() {
        const loader = document.createElement('span');
        loader.className = 'grace-loader';
        loader.innerHTML = `
            <div class="golden-aura"></div>
            <svg class="golden-ring" viewBox="0 0 50 50" xmlns="http://www.w3.org/2000/svg">
                <defs>
                    <linearGradient id="goldGradient" x1="0%" y1="0%" x2="100%" y2="100%">
                        <stop offset="0%" style="stop-color:#ffd700;stop-opacity:1" />
                        <stop offset="50%" style="stop-color:#d4af37;stop-opacity:1" />
                        <stop offset="100%" style="stop-color:#b8860b;stop-opacity:1" />
                    </linearGradient>
                    <filter id="glow">
                        <feGaussianBlur stdDeviation="2" result="coloredBlur"/>
                        <feMerge>
                            <feMergeNode in="coloredBlur"/>
                            <feMergeNode in="SourceGraphic"/>
                        </feMerge>
                    </filter>
                </defs>
                <circle class="ring-path" cx="25" cy="25" r="20"/>
                <text class="rune-glyph" x="25" y="8" text-anchor="middle" style="--flicker-duration: 2.3s">ᚱ</text>
                <text class="rune-glyph" x="42" y="25" text-anchor="middle" style="--flicker-duration: 1.8s">ᚦ</text>
                <text class="rune-glyph" x="25" y="44" text-anchor="middle" style="--flicker-duration: 2.7s">ᚾ</text>
                <text class="rune-glyph" x="8" y="25" text-anchor="middle" style="--flicker-duration: 2.1s">ᚢ</text>
                <text class="rune-glyph" x="37" y="12" text-anchor="middle" font-size="8" style="--flicker-duration: 1.5s">ᛟ</text>
                <text class="rune-glyph" x="37" y="38" text-anchor="middle" font-size="8" style="--flicker-duration: 2.9s">ᛊ</text>
                <text class="rune-glyph" x="13" y="12" text-anchor="middle" font-size="8" style="--flicker-duration: 2.2s">ᛁ</text>
                <text class="rune-glyph" x="13" y="38" text-anchor="middle" font-size="8" style="--flicker-duration: 1.7s">ᛚ</text>
            </svg>
            <div class="grace-point"></div>
        `;
        
        // Animate grace point
        const gracePoint = loader.querySelector('.grace-point');
        let pointX = 0, pointY = 0, targetX = 0, targetY = 0;
        
        function animateGracePoint() {
            targetX = (Math.random() - 0.5) * 6;
            targetY = (Math.random() - 0.5) * 6;
            pointX += (targetX - pointX) * 0.1;
            pointY += (targetY - pointY) * 0.1;
            gracePoint.style.transform = `translate(calc(-50% + ${pointX}px), calc(-50% + ${pointY}px))`;
        }
        
        gracePointInterval = setInterval(animateGracePoint, 50);
        loader.dataset.interval = gracePointInterval;
        
        return loader;
    }

    // Create simple dot loader
    function createSimpleLoader() {
        const loader = document.createElement('span');
        loader.className = 'typing-placeholder';
        return loader;
    }

    // Get appropriate loader based on current setting
    function getLoader() {
        if (currentAnimation === 'elden-ring') {
            injectEldenRingStyles();
            return createEldenRingLoader();
        }
        return createSimpleLoader();
    }

    // Clean up grace point animation
    function cleanupLoader(loader) {
        if (loader && loader.dataset.interval) {
            clearInterval(parseInt(loader.dataset.interval));
        }
    }

    // Toggle animation type
    function toggleAnimation() {
        currentAnimation = currentAnimation === 'simple' ? 'elden-ring' : 'simple';
        localStorage.setItem('loaderAnimation', currentAnimation);
        
        // Update button icon
        const btn = document.getElementById('animationToggleBtn');
        if (btn) {
            btn.innerHTML = currentAnimation === 'elden-ring' 
                ? '<span class="icon" title="Switch to simple loader">🔮</span>' 
                : '<span class="icon" title="Switch to Elden Ring loader">⭕</span>';
        }
        
        // Replace existing loaders
        document.querySelectorAll('.typing-placeholder, .grace-loader').forEach(oldLoader => {
            const parent = oldLoader.parentElement;
            if (parent) {
                cleanupLoader(oldLoader);
                const newLoader = getLoader();
                parent.replaceChild(newLoader, oldLoader);
            }
        });
        
        return currentAnimation;
    }

    // Initialize
    function init() {
        // Set initial button state
        const btn = document.getElementById('animationToggleBtn');
        if (btn) {
            btn.innerHTML = currentAnimation === 'elden-ring' 
                ? '<span class="icon" title="Switch to simple loader">🔮</span>' 
                : '<span class="icon" title="Switch to Elden Ring loader">⭕</span>';
        }
    }

    return {
        getLoader,
        toggleAnimation,
        cleanupLoader,
        init,
        getCurrentAnimation: () => currentAnimation
    };
})();

// Export for use in main script
window.AnimationManager = AnimationManager;