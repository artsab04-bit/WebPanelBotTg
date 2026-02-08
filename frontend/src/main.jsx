import React from 'react'
import { createRoot } from 'react-dom/client'
import './styles.css'

function App() {
  return (
    <div className="app">
      <header>Панель поддержки Hyper Collision</header>
      <main className="layout">
        <section className="col tickets">
          <h2>Тикеты</h2>
          <input placeholder="Поиск по ID, пользователю, сообщению" />
          <div className="filters">
            <button>Открытые</button>
            <button>Закрытые</button>
            <button>Архивные</button>
          </div>
          <ul><li>Список тикетов (realtime)</li></ul>
        </section>
        <section className="col chat">
          <h2>Чат</h2>
          <div className="messages">Сообщения тикета</div>
          <div className="composer">
            <textarea placeholder="Введите ответ" />
            <div>
              <button>Шаблоны</button>
              <button>Быстрые кнопки</button>
              <button className="accent">Отправить</button>
            </div>
          </div>
        </section>
        <section className="col info">
          <h2>Инфо-панель</h2>
          <p>Пользователь, участники, заметка, метрики</p>
          <label><input type="checkbox" /> Требует возврат средств</label>
          <div>
            <button>Закрыть тикет</button>
            <button>Открыть снова</button>
          </div>
        </section>
      </main>
    </div>
  )
}

createRoot(document.getElementById('root')).render(<App />)
