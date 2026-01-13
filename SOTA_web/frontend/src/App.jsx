import { useState } from 'react';
import InventoryVerification from './components/InventoryVerification';

function App() {
  const [image, setImage] = useState(null);
  const [imageUrl, setImageUrl] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    setImage(file);
    setImageUrl(URL.createObjectURL(file));
    setLoading(true);

    const formData = new FormData();
    formData.append('image', file);

    try {
      const res = await fetch('http://localhost:5000/api/predict', {
        method: 'POST',
        body: formData
      });
      const data = await res.json();
      setResult(data);
    } catch (err) {
      console.error('예측 실패:', err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app">
      <h1>인벤토리 분석</h1>
      
      <input type="file" accept="image/*" onChange={handleUpload} />
      
      {loading && <p>분석 중...</p>}
      
      {result && imageUrl && (
        <InventoryVerification
          screenshotUrl={imageUrl}
          slots={result.slots}
          allItems={result.all_items}
          onVerified={(labels) => console.log('검증 완료:', labels)}
        />
      )}
    </div>
  );
}

export default App;