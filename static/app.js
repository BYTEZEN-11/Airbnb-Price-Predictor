// Additional interactive features for enhanced user experience

// ============= SUGGESTIONS SIDEBAR FUNCTIONALITY =============
function toggleSidebar() {
    const sidebar = document.getElementById('suggestions-sidebar');
    const mainContent = document.querySelector('.main-content-wrapper');
    const fab = document.getElementById('suggestions-fab');
    
    sidebar.classList.toggle('active');
    
    // Update FAB icon
    if (sidebar.classList.contains('active')) {
        fab.innerHTML = '<i class="fas fa-times"></i><span class="fab-tooltip">Close Tips</span>';
    } else {
        fab.innerHTML = '<i class="fas fa-lightbulb"></i><span class="fab-tooltip">Get Smart Tips</span>';
    }
}

// Auto-highlight relevant tips based on current form section
function highlightRelevantTips() {
    const formSections = document.querySelectorAll('.form-section');
    const tipSections = document.querySelectorAll('.tip-section');
    
    // Find which form section is most visible
    let mostVisibleSection = null;
    let maxVisibleHeight = 0;
    
    formSections.forEach(section => {
        const rect = section.getBoundingClientRect();
        const visibleHeight = Math.max(0, Math.min(rect.bottom, window.innerHeight) - Math.max(rect.top, 0));
        
        if (visibleHeight > maxVisibleHeight) {
            maxVisibleHeight = visibleHeight;
            mostVisibleSection = section.getAttribute('data-section');
        }
    });
    
    // Highlight corresponding tip section
    tipSections.forEach(tipSection => {
        tipSection.classList.remove('active');
        if (tipSection.getAttribute('data-section') === mostVisibleSection) {
            tipSection.classList.add('active');
        }
    });
}

// Auto-scroll tips based on form focus
function setupTipAutoScroll() {
    const formInputs = document.querySelectorAll('.form-control');
    
    formInputs.forEach(input => {
        input.addEventListener('focus', () => {
            const formSection = input.closest('.form-section');
            if (formSection) {
                const sectionType = formSection.getAttribute('data-section');
                const correspondingTip = document.querySelector(`.tip-section[data-section="${sectionType}"]`);
                
                if (correspondingTip) {
                    correspondingTip.scrollIntoView({ 
                        behavior: 'smooth', 
                        block: 'center' 
                    });
                    
                    // Highlight the tip section temporarily
                    correspondingTip.style.backgroundColor = 'var(--accent-light)';
                    setTimeout(() => {
                        correspondingTip.style.backgroundColor = '';
                    }, 2000);
                }
            }
        });
    });
}

// Initialize sidebar functionality
document.addEventListener('DOMContentLoaded', function() {
    // Check screen size and show sidebar on desktop
    function handleResponsiveDesign() {
        const sidebar = document.getElementById('suggestions-sidebar');
        const fab = document.getElementById('suggestions-fab');
        
        if (window.innerWidth >= 1401) {
            // Desktop - show sidebar by default
            sidebar.classList.add('active');
            fab.style.display = 'none';
        } else {
            // Mobile/Tablet - hide sidebar, show FAB
            sidebar.classList.remove('active');
            fab.style.display = 'flex';
        }
    }
    
    // Initial setup
    handleResponsiveDesign();
    setupTipAutoScroll();
    
    // Handle window resize
    window.addEventListener('resize', handleResponsiveDesign);
    
    // Handle scroll for tip highlighting
    window.addEventListener('scroll', highlightRelevantTips);
    
    // Initial highlight
    setTimeout(highlightRelevantTips, 500);
});

// Add number input controls
document.addEventListener('DOMContentLoaded', function() {
    // Add increment/decrement buttons to number inputs
    const numberInputs = document.querySelectorAll('input[type="number"]');
    
    numberInputs.forEach(input => {
        const wrapper = input.closest('.input-wrapper');
        if (!wrapper) return;
        
        // Create control buttons
        const controls = document.createElement('div');
        controls.className = 'number-controls';
        controls.innerHTML = `
            <button type="button" class="num-btn num-increment" aria-label="Increase">
                <i class="fas fa-plus"></i>
            </button>
            <button type="button" class="num-btn num-decrement" aria-label="Decrease">
                <i class="fas fa-minus"></i>
            </button>
        `;
        
        wrapper.appendChild(controls);
        
        // Add event listeners
        const increment = controls.querySelector('.num-increment');
        const decrement = controls.querySelector('.num-decrement');
        
        increment.addEventListener('click', () => {
            const step = parseFloat(input.step) || 1;
            const max = parseFloat(input.max);
            const currentValue = parseFloat(input.value) || 0;
            const newValue = currentValue + step;
            
            if (!max || newValue <= max) {
                input.value = newValue;
                input.dispatchEvent(new Event('input', { bubbles: true }));
                input.dispatchEvent(new Event('change', { bubbles: true }));
            }
        });
        
        decrement.addEventListener('click', () => {
            const step = parseFloat(input.step) || 1;
            const min = parseFloat(input.min) || 0;
            const currentValue = parseFloat(input.value) || 0;
            const newValue = currentValue - step;
            
            if (newValue >= min) {
                input.value = newValue;
                input.dispatchEvent(new Event('input', { bubbles: true }));
                input.dispatchEvent(new Event('change', { bubbles: true }));
            }
        });
    });
});

// Add smooth scrolling to form sections
function scrollToSection(sectionId) {
    const section = document.querySelector(`[data-section="${sectionId}"]`);
    if (section) {
        section.scrollIntoView({
            behavior: 'smooth',
            block: 'center'
        });
        
        // Add highlight effect
        section.style.boxShadow = '0 0 0 3px var(--accent-light)';
        setTimeout(() => {
            section.style.boxShadow = '';
        }, 2000);
    }
}

// Add form completion checker
function checkFormCompletion() {
    const form = document.getElementById('prediction-form');
    const requiredFields = form.querySelectorAll('[required]');
    const completedFields = Array.from(requiredFields).filter(field => field.value.trim() !== '');
    
    const sections = document.querySelectorAll('.form-section');
    sections.forEach(section => {
        const sectionFields = section.querySelectorAll('[required]');
        const sectionCompleted = Array.from(sectionFields).every(field => field.value.trim() !== '');
        
        const sectionIcon = section.querySelector('.section-icon');
        if (sectionCompleted) {
            sectionIcon.innerHTML = '<i class="fas fa-check"></i>';
            sectionIcon.style.background = 'linear-gradient(135deg, var(--success), #059669)';
        } else {
            // Reset to original icon based on section
            const sectionType = section.getAttribute('data-section');
            const icons = {
                'property': 'fa-building',
                'space': 'fa-bed',
                'policies': 'fa-clipboard-list',
                'host': 'fa-user-tie',
                'location': 'fa-map-marked-alt'
            };
            sectionIcon.innerHTML = `<i class="fas ${icons[sectionType] || 'fa-cog'}"></i>`;
            sectionIcon.style.background = 'linear-gradient(135deg, var(--accent), var(--accent-2))';
        }
    });
    
    return completedFields.length === requiredFields.length;
}

// Add keyboard shortcuts
document.addEventListener('keydown', function(e) {
    // Ctrl/Cmd + Enter to submit form
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        e.preventDefault();
        const form = document.getElementById('prediction-form');
        if (checkFormCompletion()) {
            form.requestSubmit();
        } else {
            showNotification('Please complete all required fields before submitting.', 'warning');
        }
    }
    
    // Escape to close notifications
    if (e.key === 'Escape') {
        const notifications = document.querySelectorAll('.notification');
        notifications.forEach(notification => {
            notification.classList.remove('show');
            setTimeout(() => {
                if (notification.parentNode) {
                    notification.parentNode.removeChild(notification);
                }
            }, 300);
        });
    }
});

// Add form auto-save with visual indicator
let autoSaveTimer;
function setupAutoSave() {
    const form = document.getElementById('prediction-form');
    const indicator = document.createElement('div');
    indicator.className = 'auto-save-indicator';
    indicator.innerHTML = '<i class="fas fa-save"></i> <span>Saved</span>';
    document.body.appendChild(indicator);
    
    form.addEventListener('input', function() {
        clearTimeout(autoSaveTimer);
        indicator.innerHTML = '<i class="fas fa-clock"></i> <span>Saving...</span>';
        indicator.classList.add('saving');
        
        autoSaveTimer = setTimeout(() => {
            // Auto-save logic (already implemented in main script)
            indicator.innerHTML = '<i class="fas fa-check"></i> <span>Saved</span>';
            indicator.classList.remove('saving');
            indicator.classList.add('saved');
            
            setTimeout(() => {
                indicator.classList.remove('saved');
            }, 2000);
        }, 1000);
    });
}

// Initialize additional features
document.addEventListener('DOMContentLoaded', function() {
    setupAutoSave();
    
    // Add form completion checker to all form changes
    const form = document.getElementById('prediction-form');
    form.addEventListener('input', checkFormCompletion);
    form.addEventListener('change', checkFormCompletion);
    
    // Initial check
    setTimeout(checkFormCompletion, 500);
});