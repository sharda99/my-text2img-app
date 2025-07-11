document.addEventListener('DOMContentLoaded', () => {
  const btn     = document.getElementById('launch-btn');
  const section = document.getElementById('studio');

  btn.addEventListener('click', () => {
    section.scrollIntoView({ behavior: 'smooth' });
  });
});
