const REFRESH_INTERVAL = 500;
const AGENTS = ["Agent1", "Agent2", "Agent3"];

setInterval(async () => {
  const actions = ['buy', 'sell', 'hold'];
  const agentsData = AGENTS.map(agent => ({
    agent: agent,
    amount: Math.floor(Math.random() * 50),
    action: actions[Math.floor(Math.random() * actions.length)]
  }));

  try {
    const response = await fetch('http://localhost:8000/api/post-trades/', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        company: 'Xiaomi',
        current_price: +(Math.random() * 100).toFixed(2),
        date: new Date().toISOString(),
        agents: agentsData
      })
    });

    const data = await response.json();
    console.log('Ответ сервера:', data);
  } catch (error) {
    console.error('Ошибка при отправке запроса:', error);
  }
}, REFRESH_INTERVAL);
