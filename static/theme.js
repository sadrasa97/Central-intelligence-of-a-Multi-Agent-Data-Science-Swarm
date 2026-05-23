/**
 * Theme Management System for Dr. Data Platform
 */

class ThemeManager {
    constructor() {
        this.currentTheme = localStorage.getItem('theme') || 'dark';
        this.init();
    }

    init() {
        this.applyTheme(this.currentTheme);
        this.createToggleButton();
    }

    createToggleButton() {
        const header = document.querySelector('aside .p-6');
        if (!header) return;

        const toggleContainer = document.createElement('div');
        toggleContainer.className = 'mt-4';
        toggleContainer.innerHTML = `
            <button id="themeToggle" class="w-full flex items-center justify-between p-2 rounded-lg hover:bg-white/10 transition-all">
                <span class="text-sm">Theme</span>
                <i class="fas fa-${this.currentTheme === 'dark' ? 'moon' : 'sun'}"></i>
            </button>
        `;
        header.appendChild(toggleContainer);

        document.getElementById('themeToggle').addEventListener('click', () => this.toggleTheme());
    }

    toggleTheme() {
        this.currentTheme = this.currentTheme === 'dark' ? 'light' : 'dark';
        this.applyTheme(this.currentTheme);
        localStorage.setItem('theme', this.currentTheme);
    }

    applyTheme(theme) {
        const root = document.documentElement;
        
        if (theme === 'light') {
            // Light theme
            root.style.setProperty('--bg-primary', '#f9fafb');
            root.style.setProperty('--bg-secondary', '#ffffff');
            root.style.setProperty('--text-primary', '#1f2937');
            root.style.setProperty('--text-secondary', '#6b7280');
            root.style.setProperty('--border-color', '#e5e7eb');
            root.style.setProperty('--card-bg', '#ffffff');
            root.style.setProperty('--input-bg', '#ffffff');
            root.style.setProperty('--input-text', '#1f2937');
            root.style.setProperty('--input-border', '#d1d5db');
            
            document.body.style.background = 'linear-gradient(135deg, #f0f9ff 0%, #e0e7ff 50%, #ede9fe 100%)';
            document.body.style.color = '#1f2937';
            
            // Update glass effects
            document.querySelectorAll('.glass, .glass-card').forEach(el => {
                el.style.background = 'rgba(255, 255, 255, 0.95)';
                el.style.borderColor = 'rgba(0, 0, 0, 0.1)';
                el.style.color = '#1f2937';
            });
            
            document.querySelector('aside').style.background = 'rgba(255, 255, 255, 0.95)';
            document.querySelector('aside').style.color = '#1f2937';
            
            // Update all text elements
            document.querySelectorAll('.text-gray-400').forEach(el => {
                el.classList.remove('text-gray-400');
                el.classList.add('text-gray-600');
            });
            
        } else {
            // Dark theme
            root.style.setProperty('--bg-primary', '#0f0c29');
            root.style.setProperty('--bg-secondary', '#1a1625');
            root.style.setProperty('--text-primary', '#ffffff');
            root.style.setProperty('--text-secondary', '#94a3b8');
            root.style.setProperty('--border-color', 'rgba(255, 255, 255, 0.1)');
            root.style.setProperty('--card-bg', 'rgba(255, 255, 255, 0.07)');
            root.style.setProperty('--input-bg', '#ffffff');
            root.style.setProperty('--input-text', '#1f2937');
            root.style.setProperty('--input-border', 'rgba(255, 255, 255, 0.2)');
            
            document.body.style.background = 'linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%)';
            document.body.style.color = '#ffffff';
            
            // Reset glass effects
            document.querySelectorAll('.glass, .glass-card').forEach(el => {
                el.style.background = 'rgba(255, 255, 255, 0.07)';
                el.style.borderColor = 'rgba(255, 255, 255, 0.15)';
                el.style.color = '#ffffff';
            });
            
            document.querySelector('aside').style.background = 'rgba(255, 255, 255, 0.05)';
            document.querySelector('aside').style.color = '#ffffff';
            
            // Update text colors back
            document.querySelectorAll('.text-gray-600').forEach(el => {
                el.classList.remove('text-gray-600');
                el.classList.add('text-gray-400');
            });
        }
        
        // Update icon
        const icon = document.querySelector('#themeToggle i');
        if (icon) {
            icon.className = `fas fa-${theme === 'dark' ? 'moon' : 'sun'}`;
        }
        
        // Ensure inputs remain readable
        document.querySelectorAll('input, select, textarea').forEach(el => {
            if (el.type !== 'range' && el.type !== 'checkbox' && el.type !== 'radio') {
                el.style.backgroundColor = '#ffffff';
                el.style.color = '#1f2937';
                el.style.border = '1px solid #d1d5db';
            }
        });
        
        document.querySelectorAll('option').forEach(el => {
            el.style.backgroundColor = '#ffffff';
            el.style.color = '#1f2937';
        });
    }
}

// Initialize theme manager when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => new ThemeManager());
} else {
    new ThemeManager();
}
