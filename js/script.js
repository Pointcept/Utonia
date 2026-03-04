// Interactive video examples functionality
document.addEventListener('DOMContentLoaded', function() {
    const thumbnails = document.querySelectorAll('.thumbnail');
    const videoPlayer = document.getElementById('examples-video-player');
    
    thumbnails.forEach(thumbnail => {
        thumbnail.addEventListener('click', function() {
            const videoSrc = this.getAttribute('data-video');
            
            // Update video source
            videoPlayer.src = videoSrc;
            
            // Remove active class from all thumbnails
            thumbnails.forEach(thumb => thumb.classList.remove('active'));
            
            // Add active class to clicked thumbnail
            this.classList.add('active');
            
            // Play the video
            videoPlayer.load();
            videoPlayer.play();
        });
    });
    
    // Set first thumbnail as active by default
    if (thumbnails.length > 0) {
        thumbnails[0].classList.add('active');
    }
});

// Smooth scrolling for navigation links
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
            target.scrollIntoView({
                behavior: 'smooth',
                block: 'start'
            });
        }
    });
});

// Add scroll effect to navbar
window.addEventListener('scroll', function() {
    const navbar = document.querySelector('.navbar');
    if (window.scrollY > 100) {
        navbar.style.background = 'rgba(255, 255, 255, 0.98)';
        navbar.style.boxShadow = '0 2px 20px rgba(0, 0, 0, 0.1)';
    } else {
        navbar.style.background = 'rgba(255, 255, 255, 0.95)';
        navbar.style.boxShadow = 'none';
    }
});

// Add loading animation for images
document.addEventListener('DOMContentLoaded', function() {
    const images = document.querySelectorAll('img');
    
    images.forEach(img => {
        img.addEventListener('load', function() {
            this.style.opacity = '1';
        });
        
        img.style.opacity = '0';
        img.style.transition = 'opacity 0.3s ease';
    });
});

// Add intersection observer for animations
const observerOptions = {
    threshold: 0.1,
    rootMargin: '0px 0px -50px 0px'
};

const observer = new IntersectionObserver(function(entries) {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            entry.target.style.opacity = '1';
            entry.target.style.transform = 'translateY(0)';
        }
    });
}, observerOptions);

// Observe all sections for animation
document.querySelectorAll('section').forEach(section => {
    section.style.opacity = '0';
    section.style.transform = 'translateY(30px)';
    section.style.transition = 'opacity 0.8s ease, transform 0.8s ease';
    observer.observe(section);
});

// Smooth scrolling for navigation links
document.addEventListener('DOMContentLoaded', function() {
    // Smooth scrolling for navigation links
    const navLinks = document.querySelectorAll('.nav-menu a[href^="#"]');
    
    navLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            
            const targetId = this.getAttribute('href');
            const targetSection = document.querySelector(targetId);
            
            if (targetSection) {
                const offsetTop = targetSection.offsetTop - 80; // Account for fixed header
                
                window.scrollTo({
                    top: offsetTop,
                    behavior: 'smooth'
                });
            }
        });
    });

    // Add active class to navigation links on scroll
    const sections = document.querySelectorAll('section[id]');
    const navItems = document.querySelectorAll('.nav-menu a');

    window.addEventListener('scroll', () => {
        let current = '';
        
        sections.forEach(section => {
            const sectionTop = section.offsetTop - 100;
            const sectionHeight = section.clientHeight;
            
            if (window.pageYOffset >= sectionTop && window.pageYOffset < sectionTop + sectionHeight) {
                current = section.getAttribute('id');
            }
        });

        navItems.forEach(item => {
            item.classList.remove('active');
            if (item.getAttribute('href') === `#${current}`) {
                item.classList.add('active');
            }
        });
    });

    // Add hover effects for buttons
    const buttons = document.querySelectorAll('.btn');
    buttons.forEach(button => {
        button.addEventListener('mouseenter', function() {
            this.style.transform = 'translateY(-2px)';
        });
        
        button.addEventListener('mouseleave', function() {
            this.style.transform = 'translateY(0)';
        });
    });

    // Lazy loading for images
    const images = document.querySelectorAll('img');
    const imageObserver = new IntersectionObserver((entries, observer) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const img = entry.target;
                img.style.opacity = '1';
                img.style.transform = 'translateY(0)';
                observer.unobserve(img);
            }
        });
    });

    images.forEach(img => {
        img.style.opacity = '0';
        img.style.transform = 'translateY(20px)';
        img.style.transition = 'opacity 0.6s ease, transform 0.6s ease';
        imageObserver.observe(img);
    });

    // Add copy functionality to citation
    const citationBox = document.querySelector('.citation-box');
    if (citationBox) {
        citationBox.addEventListener('click', function() {
            const text = this.querySelector('code').textContent;
            navigator.clipboard.writeText(text).then(() => {
                // Show a temporary "Copied!" message
                const originalContent = this.innerHTML;
                this.innerHTML = '<div style="text-align: center; color: #10b981; font-size: 1.1rem; padding: 2rem;">Copied to clipboard!</div>';
                
                setTimeout(() => {
                    this.innerHTML = originalContent;
                }, 2000);
            });
        });
        
        citationBox.style.cursor = 'pointer';
        citationBox.title = 'Click to copy citation';
    }
});

// Add CSS for active navigation state
const style = document.createElement('style');
style.textContent = `
    .nav-menu a.active {
        color: #2563eb !important;
        font-weight: 600;
    }
    
    .citation-box:hover {
        transform: scale(1.02);
        transition: transform 0.3s ease;
    }
`;
document.head.appendChild(style);

// Lightbox for Gallery
const lightboxModal = document.getElementById('lightbox-modal');
const lightboxImg = document.getElementById('lightbox-img');
const lightboxClose = document.getElementById('lightbox-close');

// Add event listeners for all lightbox triggers (gallery and repainting)
document.querySelectorAll('.gallery-img, .lightbox-trigger').forEach(el => {
    el.addEventListener('click', function(e) {
        e.preventDefault();
        const fullImg = this.getAttribute('data-full') || this.querySelector('img')?.getAttribute('src');
        if (fullImg) {
            lightboxImg.src = fullImg;
            lightboxModal.style.display = 'block';
        }
    });
});

// Close lightbox
lightboxClose.addEventListener('click', function() {
    lightboxModal.style.display = 'none';
    lightboxImg.src = '';
});
lightboxModal.addEventListener('click', function(e) {
    if (e.target === lightboxModal) {
        lightboxModal.style.display = 'none';
        lightboxImg.src = '';
    }
});

// === Dynamic Demo Examples Thumbnails and Video Player ===
document.addEventListener('DOMContentLoaded', function() {
    const videoFiles = [
        { video: 'video_demo/example_1-1.mp4', thumb: 'thumbnails/example_1_thumb.jpg' },
        { video: 'video_demo/example_2-1.mp4', thumb: 'thumbnails/example_2_thumb.jpg' },
        { video: 'video_demo/example_3-1.mp4', thumb: 'thumbnails/example_4_thumb.jpg' },
        { video: 'video_demo/example_4-1.mp4', thumb: 'thumbnails/example_4_thumb.jpg' },
        { video: 'video_demo/example_5-1.mp4', thumb: 'thumbnails/example_5_thumb.jpg' },
        { video: 'video_demo/example_6-1.mp4', thumb: 'thumbnails/example_6_thumb.jpg' },
        { video: 'video_demo/example_7-1.mp4', thumb: 'thumbnails/example_7_thumb.jpg' },
        { video: 'video_demo/example_8-1.mp4', thumb: 'thumbnails/example_8_thumb.jpg' },
    ];
    const thumbnailsContainer = document.getElementById('examples-thumbnails');
    const videoPlayer = document.getElementById('examples-video-player');

    // Center the thumbnails row
    thumbnailsContainer.style.display = 'flex';
    thumbnailsContainer.style.justifyContent = 'center';
    thumbnailsContainer.style.alignItems = 'center';
    thumbnailsContainer.style.gap = '16px';
    thumbnailsContainer.style.marginBottom = '20px';

    function createThumbnail(videoObj, idx) {
        const thumbDiv = document.createElement('div');
        thumbDiv.className = 'thumbnail';
        thumbDiv.style.cursor = 'pointer';
        thumbDiv.style.borderRadius = '8px';
        thumbDiv.style.overflow = 'hidden';
        thumbDiv.style.border = '2px solid transparent';
        thumbDiv.style.width = '90px';
        thumbDiv.style.height = '60px';
        thumbDiv.style.position = 'relative';
        thumbDiv.style.background = '#eee';

        // Use static image for thumbnail
        const img = document.createElement('img');
        img.src = videoObj.thumb;
        img.alt = 'Video thumbnail';
        img.style.width = '100%';
        img.style.height = '100%';
        img.style.objectFit = 'cover';
        thumbDiv.appendChild(img);

        // Click event
        thumbDiv.addEventListener('click', function() {
            // Remove active from all
            document.querySelectorAll('.thumbnail').forEach(t => t.style.border = '2px solid transparent');
            thumbDiv.style.border = '2px solid #2563eb';
            // Set video
            videoPlayer.src = videoObj.video;
            videoPlayer.load();
            videoPlayer.play();
        });

        // Set first as active
        if (idx === 0) {
            setTimeout(() => {
                thumbDiv.style.border = '2px solid #2563eb';
                videoPlayer.src = videoObj.video;
                videoPlayer.load();
            }, 100);
        }

        thumbnailsContainer.appendChild(thumbDiv);
    }

    // Generate thumbnails
    videoFiles.forEach((obj, idx) => createThumbnail(obj, idx));
});

// === Applications Interactive Gallery ===
const appVideos = [
    { src: 'application_1_resized.mp4', caption: 'Relighting', thumb: 'thumbnails/example_1_thumb.jpg' },
    { src: 'application_2.mp4', caption: 'Interactable Scene', thumb: 'thumbnails/example_2_thumb.jpg' },
    { src: 'application_3_resized.mp4', caption: 'Physically Based Interaction', thumb: 'thumbnails/example_4_thumb.jpg' }
];
const appThumbs = document.getElementById('applications-thumbnails');
const appPlayer = document.getElementById('applications-video-player');
const appCaption = document.getElementById('applications-caption');

if (appThumbs && appPlayer && appCaption) {
    appVideos.forEach((vid, idx) => {
        // Create thumbnail from static image
        const thumbDiv = document.createElement('div');
        thumbDiv.className = 'app-thumbnail';
        thumbDiv.style.cursor = 'pointer';
        thumbDiv.style.borderRadius = '8px';
        thumbDiv.style.overflow = 'hidden';
        thumbDiv.style.border = '2px solid transparent';
        thumbDiv.style.width = '120px';
        thumbDiv.style.height = '80px';
        thumbDiv.style.position = 'relative';
        thumbDiv.style.background = '#eee';

        // Use static image for thumbnail
        const img = document.createElement('img');
        img.src = vid.thumb;
        img.alt = 'Application video thumbnail';
        img.style.width = '100%';
        img.style.height = '100%';
        img.style.objectFit = 'cover';
        thumbDiv.appendChild(img);

        thumbDiv.addEventListener('click', function() {
            document.querySelectorAll('.app-thumbnail').forEach(t => t.style.border = '2px solid transparent');
            thumbDiv.style.border = '2px solid #2563eb';
            appPlayer.src = vid.src;
            appPlayer.load();
            appPlayer.play();
            appCaption.textContent = vid.caption;
        });

        // Set first as active
        if (idx === 0) {
            setTimeout(() => {
                thumbDiv.style.border = '2px solid #2563eb';
                appPlayer.src = vid.src;
                appPlayer.load();
                appCaption.textContent = vid.caption;
            }, 100);
        }

        appThumbs.appendChild(thumbDiv);
    });
} 