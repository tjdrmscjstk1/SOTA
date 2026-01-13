import { useState } from 'react';
import './InventoryVerification.css';

function InventoryVerification({ screenshotUrl, slots, allItems, onVerified }) {
  const [items, setItems] = useState(slots);
  const [selectedSlot, setSelectedSlot] = useState(null);
  const [searchTerm, setSearchTerm] = useState('');

  const getAssetUrl = (label) => 
    `http://localhost:5000/api/assets/artifacts/${label}.png`;

  const updateItem = (idx, newLabel) => {
    setItems(prev => prev.map((item, i) =>
      i === idx ? { ...item, prediction: newLabel, corrected: true } : item
    ));
    setSelectedSlot(null);
  };

  const filteredItems = allItems.filter(item =>
    item.toLowerCase().includes(searchTerm.toLowerCase())
  );

  return (
    <div className="verification-container">
      <div className="screenshot-wrapper">
        <img src={screenshotUrl} alt="screenshot" />
        
        {items.map((slot, idx) => (
          <div
            key={idx}
            className={`slot-overlay ${slot.corrected ? 'corrected' : ''}`}
            style={{
              left: slot.bbox.x,
              top: slot.bbox.y,
              width: slot.bbox.w,
              height: slot.bbox.h,
              border: slot.confidence < 0.7 ? '2px solid red' : '2px solid lime'
            }}
            onClick={() => setSelectedSlot(idx)}
          >
            <img src={getAssetUrl(slot.prediction)} alt={slot.prediction} />
          </div>
        ))}
      </div>

      {/* 수정 모달 */}
      {selectedSlot !== null && (
        <div className="modal-overlay" onClick={() => setSelectedSlot(null)}>
          <div className="modal" onClick={e => e.stopPropagation()}>
            <h3>슬롯 {selectedSlot + 1}</h3>
            
            <p>현재: {items[selectedSlot].prediction} 
              ({(items[selectedSlot].confidence * 100).toFixed(0)}%)</p>
            
            {/* Top-3 */}
            <div className="top3">
              {items[selectedSlot].top3.map((label, i) => (
                <button key={i} onClick={() => updateItem(selectedSlot, label)}>
                  <img src={getAssetUrl(label)} alt={label} />
                  <span>{label}</span>
                </button>
              ))}
            </div>
            
            {/* 검색 */}
            <input
              type="text"
              placeholder="아이템 검색..."
              value={searchTerm}
              onChange={e => setSearchTerm(e.target.value)}
            />
            
            <div className="item-grid">
              {filteredItems.slice(0, 20).map(label => (
                <button key={label} onClick={() => updateItem(selectedSlot, label)}>
                  <img src={getAssetUrl(label)} alt={label} />
                </button>
              ))}
            </div>
            
            <button className="confirm-btn" onClick={() => setSelectedSlot(null)}>
              ✓ 확인
            </button>
          </div>
        </div>
      )}

      <button onClick={() => onVerified(items.map(i => i.prediction))}>
        검증 완료
      </button>
    </div>
  );
}

export default InventoryVerification;