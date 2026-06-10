const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

let isWaiting = false;
let pendingImageBase64 = null;
let pendingImageName = null;

const messagesContainer = $('#messages');
const welcomeScreen = $('#welcome');
const queryInput = $('#query-input');
const btnSend = $('#btn-send');
const btnUpload = $('#btn-upload');
const fileInput = $('#file-input');
const imgPreviewBar = $('#img-preview-bar');
const imgPreviewName = $('#img-preview-name');
const imgPreviewSize = $('#img-preview-size');
const btnRemoveImg = $('#btn-remove-img');
const statusDot = $('#status-dot');
const statusText = $('#status-text');
const imageGallery = $('#image-gallery');
const galleryGrid = $('#gallery-grid');
const temarioPanel = $('#temario-panel');
const temarioList = $('#temario-list');
const lightbox = $('#lightbox');
const lightboxImg = $('#lightbox-img');
const lightboxCaption = $('#lightbox-caption');

function formatSize(bytes) {
  if (!bytes) return '';
  if (bytes < 1024) return bytes + ' B';
  if (bytes < 1048576) return (bytes / 1024).toFixed(1) + ' KB';
  return (bytes / 1048576).toFixed(1) + ' MB';
}

function renderMarkdown(text) {
  if (!text) return '';
  let html = text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/^### (.+)$/gm, '<h3>$1</h3>')
    .replace(/^## (.+)$/gm, '<h2>$1</h2>')
    .replace(/^# (.+)$/gm, '<h1>$1</h1>')
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/\*(.+?)\*/g, '<em>$1</em>')
    .replace(/`([^`]+)`/g, '<code>$1</code>')
    .replace(/^- (.+)$/gm, '<li>$1</li>')
    .replace(/(<li>.*<\/li>\n?)+/g, '<ul>$&</ul>')
    .replace(/\n/g, '<br>');
  return html;
}

function addMessage(role, content, images = []) {
  welcomeScreen.style.display = 'none';
  const div = document.createElement('div');
  div.className = 'message ' + role;
  
  let innerContent = role === 'assistant' || role === 'error' ? renderMarkdown(content) : content;
  
  if (images && images.length > 0) {
    let imagesHtml = '<div class="msg-images">';
    images.forEach(img => {
      if (img.isUserUpload && img.base64) {
        imagesHtml += `<img src="data:image/png;base64,${img.base64}" class="chat-uploaded-image" onclick="openLightbox(this.src, '')" />`;
      }
    });
    imagesHtml += '</div>';
    innerContent += imagesHtml;
  }
  
  div.innerHTML = '<div class="msg-content">' + innerContent + '</div>';
  messagesContainer.appendChild(div);
  messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

function showGallery(images) {
  if (!images || images.length === 0) {
    imageGallery.style.display = 'none';
    return;
  }
  galleryGrid.innerHTML = '';
  images.forEach((img, i) => {
    const card = document.createElement('div');
    card.className = 'gallery-card';
    const imgEl = document.createElement('img');
    imgEl.src = img.base64 ? 'data:image/png;base64,' + img.base64 : '/imagenes_extraidas/' + img.nombre_archivo;
    imgEl.alt = img.etiqueta || 'Imagen ' + (i + 1);
    imgEl.onclick = () => openLightbox(imgEl.src, img.etiqueta || img.caption);
    card.appendChild(imgEl);
    if (img.etiqueta) {
      const lbl = document.createElement('span');
      lbl.className = 'gallery-label';
      lbl.textContent = img.etiqueta;
      card.appendChild(lbl);
    }
    if (img.caption) {
      const cap = document.createElement('span');
      cap.className = 'gallery-caption';
      cap.textContent = img.caption.substring(0, 80);
      card.appendChild(cap);
    }
    galleryGrid.appendChild(card);
  });
  imageGallery.style.display = 'block';
}

function openLightbox(src, caption) {
  lightboxImg.src = src;
  lightboxCaption.textContent = caption || '';
  lightbox.style.display = 'flex';
}

function closeLightbox() {
  lightbox.style.display = 'none';
}

lightbox.querySelector('.lightbox-close').onclick = closeLightbox;
lightbox.onclick = (e) => { if (e.target === lightbox) closeLightbox(); };
document.addEventListener('keydown', (e) => { if (e.key === 'Escape') closeLightbox(); });

async function sendMessage() {
  const query = queryInput.value.trim();
  if (!query) return;
  if (isWaiting) return;

  isWaiting = true;
  btnSend.disabled = true;
  addMessage('user', query, pendingImageBase64 ? [{base64: pendingImageBase64, isUserUpload: true}] : []);
  queryInput.value = '';
  imgPreviewBar.style.display = 'none';

  const loadingDiv = document.createElement('div');
  loadingDiv.className = 'message assistant loading';
  loadingDiv.innerHTML = '<div class="msg-content"><span class="typing-dots"><span>.</span><span>.</span><span>.</span></span></div>';
  messagesContainer.appendChild(loadingDiv);

  try {
    const body = {
      query: query,
      image_base64: pendingImageBase64 || '',
      image_filename: pendingImageName || '',
    };
    const resp = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    const data = await resp.json();
    if (resp.status >= 400) {
      loadingDiv.remove();
      addMessage('error', 'Error del servidor: ' + (data.detail || 'Ocurrió un problema inesperado.'));
      btnSend.disabled = false;
      return;
    }

    loadingDiv.remove();

    addMessage('assistant', data.respuesta, data.imagenes_recuperadas);
    if (data.mostrar_imagenes && data.imagenes_recuperadas && data.imagenes_recuperadas.length > 0) {
      showGallery(data.imagenes_recuperadas);
    }
    if (data.estructura_identificada) {
      const structDiv = document.createElement('div');
      structDiv.className = 'message assistant';
      structDiv.innerHTML = '<div class="msg-content structure-badge">Estructura identificada: <strong>' + data.estructura_identificada + '</strong></div>';
      messagesContainer.appendChild(structDiv);
    }
  } catch (err) {
    loadingDiv.remove();
    let msg = 'Error de conexión: ' + err.message;
    if (err.message.toLowerCase().includes('networkerror') || err.message.toLowerCase().includes('failed to fetch')) {
      msg = 'El servidor aún se está inicializando o no está disponible. Por favor, espera unos instantes hasta que el indicador de estado (arriba a la derecha) cambie a verde.';
    }
    addMessage('error', msg);
  }
  pendingImageBase64 = null;
  pendingImageName = null;
  isWaiting = false;
  btnSend.disabled = false;
  messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

btnSend.addEventListener('click', sendMessage);
queryInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});

btnUpload.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', (e) => {
  const file = e.target.files[0];
  if (!file) return;
  pendingImageName = file.name;
  const reader = new FileReader();
  reader.onload = (ev) => {
    pendingImageBase64 = ev.target.result.split(',')[1];
    imgPreviewName.textContent = file.name;
    imgPreviewSize.textContent = formatSize(file.size);
    imgPreviewBar.style.display = 'flex';
  };
  reader.readAsDataURL(file);
});

btnRemoveImg.addEventListener('click', () => {
  pendingImageBase64 = null;
  pendingImageName = null;
  imgPreviewBar.style.display = 'none';
  fileInput.value = '';
  fetch('/api/imagen/limpiar', { method: 'POST' });
});

$('#btn-clear-img').addEventListener('click', () => {
  fetch('/api/imagen/limpiar', { method: 'POST' });
  pendingImageBase64 = null;
  pendingImageName = null;
  imgPreviewBar.style.display = 'none';
  imageGallery.style.display = 'none';
});

$('#btn-temario').addEventListener('click', async () => {
  if (temarioPanel.style.display === 'block') {
    temarioPanel.style.display = 'none';
    return;
  }
  temarioPanel.style.display = 'block';
  const resp = await fetch('/api/temario');
  const data = await resp.json();
  temarioList.innerHTML = (data.anclas || []).map(a => '<li>' + a + '</li>').join('');
});

$('#btn-close-temario').addEventListener('click', () => {
  temarioPanel.style.display = 'none';
});

document.querySelectorAll('.chip').forEach(chip => {
  chip.addEventListener('click', () => {
    queryInput.value = chip.dataset.query;
    sendMessage();
  });
});

async function checkStatus() {
  try {
    const resp = await fetch('/api/status');
    const data = await resp.json();
    if (data.ready) {
      statusDot.className = 'status-dot online';
      statusText.textContent = 'Online — ' + data.chunks_indexed + ' chunks, ' + data.images_indexed + ' img';
    } else {
      statusDot.className = 'status-dot offline';
      statusText.textContent = 'No disponible';
    }
  } catch {
    statusDot.className = 'status-dot offline';
    statusText.textContent = 'Sin conexion';
  }
}

checkStatus();
setInterval(checkStatus, 15000);
