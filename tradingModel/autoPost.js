const REFRESH_INTERVAL = 500;

setInterval(async () => {
    const actions = ['buy', 'sell', 'hold'];
    const action = actions[Math.floor(Math.random() * actions.length)];
    try {
        const response = await fetch('http://localhost:8000/api/post-trades/', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                company: 'Xiaomi',
                current_price: Math.random() * 100,
                amount: Math.floor(Math.random() * 50),
                action: action,
                date: new Date().toISOString()
            })
        });

        const data = await response.json();
        console.log('Ответ сервера:', data);
    } catch (error) {
        console.error('Ошибка при отправке запроса:', error);
    }
}, REFRESH_INTERVAL);
