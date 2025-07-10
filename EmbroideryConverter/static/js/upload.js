// Upload functionality with loading states and validation
document.addEventListener('DOMContentLoaded', function() {
    const uploadForm = document.getElementById('uploadForm');
    const uploadBtn = document.getElementById('uploadBtn');
    const fileInput = document.getElementById('file');
    const loadingModal = new bootstrap.Modal(document.getElementById('loadingModal'));
    
    // File validation
    const allowedTypes = ['image/png', 'image/jpeg', 'image/jpg', 'image/svg+xml'];
    const maxSize = 16 * 1024 * 1024; // 16MB
    
    // File input change handler
    fileInput.addEventListener('change', function(e) {
        const file = e.target.files[0];
        if (file) {
            validateFile(file);
        }
    });
    
    // Form submission handler
    uploadForm.addEventListener('submit', function(e) {
        const file = fileInput.files[0];
        
        if (!file) {
            e.preventDefault();
            showAlert('Please select a file to upload.', 'error');
            return;
        }
        
        if (!validateFile(file)) {
            e.preventDefault();
            return;
        }
        
        // Show loading modal
        loadingModal.show();
        
        // Update button state
        uploadBtn.disabled = true;
        uploadBtn.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>Processing...';
    });
    
    // File validation function
    function validateFile(file) {
        // Check file type
        if (!allowedTypes.includes(file.type)) {
            showAlert('Invalid file type. Please upload PNG, JPG, JPEG, or SVG files only.', 'error');
            return false;
        }
        
        // Check file size
        if (file.size > maxSize) {
            showAlert('File is too large. Maximum size is 16MB.', 'error');
            return false;
        }
        
        // File is valid
        showFileInfo(file);
        return true;
    }
    
    // Show file information
    function showFileInfo(file) {
        const fileSize = formatFileSize(file.size);
        const fileType = file.type.split('/')[1].toUpperCase();
        
        // Update file input label or show info
        const fileInfoText = `Selected: ${file.name} (${fileType}, ${fileSize})`;
        
        // Create or update file info element
        let fileInfo = document.getElementById('fileInfo');
        if (!fileInfo) {
            fileInfo = document.createElement('div');
            fileInfo.id = 'fileInfo';
            fileInfo.className = 'mt-2 text-muted small';
            fileInput.parentNode.appendChild(fileInfo);
        }
        
        fileInfo.innerHTML = `<i class="fas fa-file-image me-1"></i>${fileInfoText}`;
    }
    
    // Format file size
    function formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }
    
    // Show alert messages
    function showAlert(message, type) {
        const alertClass = type === 'error' ? 'alert-danger' : 'alert-info';
        const alertHtml = `
            <div class="alert ${alertClass} alert-dismissible fade show" role="alert">
                ${message}
                <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
            </div>
        `;
        
        // Insert alert at the top of the form
        const alertContainer = document.createElement('div');
        alertContainer.innerHTML = alertHtml;
        uploadForm.parentNode.insertBefore(alertContainer.firstElementChild, uploadForm);
        
        // Auto-dismiss after 5 seconds
        setTimeout(() => {
            const alert = document.querySelector('.alert');
            if (alert) {
                const bsAlert = new bootstrap.Alert(alert);
                bsAlert.close();
            }
        }, 5000);
    }
    
    // Drag and drop functionality
    const uploadCard = document.querySelector('.card-body');
    
    uploadCard.addEventListener('dragover', function(e) {
        e.preventDefault();
        uploadCard.classList.add('dragover');
    });
    
    uploadCard.addEventListener('dragleave', function(e) {
        e.preventDefault();
        uploadCard.classList.remove('dragover');
    });
    
    uploadCard.addEventListener('drop', function(e) {
        e.preventDefault();
        uploadCard.classList.remove('dragover');
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            const file = files[0];
            
            // Update file input
            const dt = new DataTransfer();
            dt.items.add(file);
            fileInput.files = dt.files;
            
            // Validate file
            validateFile(file);
        }
    });
    
    // Progress tracking (if needed for large files)
    if (window.FormData && window.XMLHttpRequest) {
        // Enhanced upload with progress tracking could be added here
        // For now, we'll use the standard form submission
    }
    
    // Handle page unload during upload
    window.addEventListener('beforeunload', function(e) {
        if (uploadBtn.disabled) {
            e.preventDefault();
            e.returnValue = 'Upload in progress. Are you sure you want to leave?';
        }
    });
    
    // Reset form state if needed
    function resetForm() {
        uploadBtn.disabled = false;
        uploadBtn.innerHTML = '<i class="fas fa-magic me-2"></i>Generate Embroidery Pattern';
        loadingModal.hide();
        
        const fileInfo = document.getElementById('fileInfo');
        if (fileInfo) {
            fileInfo.remove();
        }
    }
    
    // Add visual feedback for file selection
    fileInput.addEventListener('focus', function() {
        this.parentNode.classList.add('focused');
    });
    
    fileInput.addEventListener('blur', function() {
        this.parentNode.classList.remove('focused');
    });
});

// Additional utility functions for enhanced UX
function showProcessingSteps() {
    const steps = [
        'Analyzing uploaded file...',
        'Converting to vector format...',
        'Generating stitch patterns...',
        'Creating embroidery preview...',
        'Optimizing colors and threads...',
        'Finalizing pattern...'
    ];
    
    let currentStep = 0;
    const stepInterval = setInterval(() => {
        if (currentStep < steps.length) {
            console.log(`Step ${currentStep + 1}: ${steps[currentStep]}`);
            currentStep++;
        } else {
            clearInterval(stepInterval);
        }
    }, 2000);
}

// Error handling for network issues
window.addEventListener('online', function() {
    showAlert('Connection restored. You can now upload files.', 'success');
});

window.addEventListener('offline', function() {
    showAlert('No internet connection. Please check your connection and try again.', 'error');
});
