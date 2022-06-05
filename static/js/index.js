const upload = document.getElementById('load');

function handleRadioClick() {
    if (document.getElementById('btnactive').checked) {
      upload.style.display = 'none';
    } else {
      upload.style.display = 'block';
    }
}

const radioButtons = document.querySelectorAll('input[name="btnradio"]');
radioButtons.forEach(radio => {
  radio.addEventListener('click', handleRadioClick);
});