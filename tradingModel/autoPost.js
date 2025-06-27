const REFRESH_INTERVAL = 50;
const AGENTS = ["Agent1", "Agent2"];

function getRandomDateWithinHours(hours = 24) {
  const now = new Date();
  const offset = Math.floor(Math.random() * hours * 60 * 60 * 1000);
  return new Date(now - offset).toISOString();
}

function getRandomAgents() {
  const shuffled = [...AGENTS].sort(() => 0.5 - Math.random());
  return shuffled.slice(0, Math.floor(Math.random() * AGENTS.length) + 1);
}

setInterval(async () => {
  const actions = ['buy', 'sell', 'hold'];
  const agentsData = getRandomAgents().map(agent => ({
    agent: agent,
    amount: Math.floor(Math.random() * 50),
    action: actions[Math.floor(Math.random() * actions.length)],
    value: +(Math.random() * 10000).toFixed(2)
  }));

  try {
    const response = await fetch('http://localhost:8000/api/post-trades/', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        company: 'Xiaomi',
        current_price: +(Math.random() * 100).toFixed(2),
        date: getRandomDateWithinHours(6),
        agents: agentsData
      })
    });

    const data = await response.json();
    console.log('Ответ сервера:', data);
  } catch (error) {
    console.error('Ошибка при отправке запроса:', error.message);
  }
}, REFRESH_INTERVAL);