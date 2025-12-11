# Smart Investment Advisor

**Smart Investment Advisor** is a research-backed, web-based tool that helps individual investors plan their finances by turning a few simple inputs into clear, inflation-aware investment projections and portfolio suggestions.

The project combines a polished landing page with an embedded Streamlit app and a flexible GenAI-powered backend to make financial planning more accessible and transparent.

[Smart Investment Advisor] - (https://smartinvestmentadvisor.netlify.app/)

<img width="1918" height="847" alt="image" src="https://github.com/user-attachments/assets/53c53325-2acd-4c3f-936a-0760fa6de01d" />  
---

### Features

- Collects a compact financial profile: **age, income, investment horizon, monthly investment, current savings, goals, and risk level**
- Infers or uses chosen risk level and maps it to an allocation across **Equity, Debt, Gold, and Cash**
- Projects future corpus using standard **time-value-of-money** formulas (lump sum + SIP-style monthly contributions, compounded monthly)
- Adjusts for inflation to show **real (today's money)** view of future wealth
- Uses **GenAI** (via OpenRouter + Gemini Flash) to generate a detailed, plain-language **personalized advisor note**
- Displays KPIs and interactive **Plotly charts** (pie for allocation, line for growth, comparison views)
- Clean, responsive **Streamlit dashboard** with tabbed results
- Responsive landing page with hero section, embedded app iframe, glossary, and footer

---

### Tech Stack

**Frontend Shell**
- `index.html`, `stylev2.css`, `main.js` – Marketing landing page and iframe embedding

**Backend App**
- **Streamlit** – UI and dashboard
- **Plotly** – Interactive charts
- **requests** – OpenRouter LLM API calls
- **numpy**, **pandas** – Financial projections and data handling

---

### Project Structure

```
smart-investment-advisor/
├── app.py                  # Main Streamlit application
├── index.html              # Landing page with embedded iframe
├── stylev2.css             # Landing page styling
├── main.js                 # Lightweight JS (iframe handling)
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

---

### Getting Started (Local Development)

```bash
# Clone the repository
git clone https://github.com/your-username/smart-investment-advisor.git
cd smart-investment-advisor

# Create and activate virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

#### Set OpenRouter API Key

```bash
# Linux / macOS
export OPENROUTER_API_KEY="your_api_key_here"

# Windows PowerShell
$env:OPENROUTER_API_KEY="your_api_key_here"
```

You can get a free key at [https://openrouter.ai](https://openrouter.ai)

#### Run the app

```bash
streamlit run app.py
```

App will be available at `http://localhost:8501`

To see the full landing page experience:
- Serve `index.html` with any static server
- Update the iframe `src` in `index.html` to your local or deployed Streamlit URL

---

### Deployment

Perfect for **Streamlit Community Cloud** (free tier works great).

1. Push your repo to GitHub
2. Connect to [share.streamlit.io](https://share.streamlit.io)
3. Add `OPENROUTER_API_KEY` in **Settings → Secrets**
4. Update `index.html` iframe to your deployed Streamlit URL

**secrets.toml** example:
```toml
OPENROUTER_API_KEY = "sk-or-..."
```

---

### How It Works

1. User lands on the marketing page or directly on the Streamlit app
2. Optional `?risk=Low|Medium|High` query parameter pre-seeds risk
3. Simple form collects financial profile and goals
4. Risk → allocation → expected return mapping
5. Future wealth projection with monthly compounding
6. GenAI generates a personalized, human-readable advisor note
7. Results shown in two tabs:
   - **Main Advisor** – Allocation, growth chart, AI note
   - **Inflation & Comparison** – Real wealth, FD vs Portfolio

---

### Important Disclaimer

> This tool is for **educational and research purposes only**.  
> Projections are based on assumptions and historical averages.  
> Past performance is not indicative of future results.  
> **This is not financial advice.** Always consult a qualified financial professional.

---

### Future Enhancements

- Stronger input validation and guardrails
- Goal-based planning (retirement, education, house purchase)
- Multiple scenario comparisons
- India-specific tax and product logic
- Multi-language support
- PDF export of results

---

### Contributing

Contributions are very welcome! Feel free to:
- Open issues
- Submit pull requests
- Improve allocation logic or LLM prompts

---

### License

[MIT License](LICENSE) – free to use, modify, and distribute.

---

**Made with ❤️ for smarter, simpler financial planning**

Give it a ⭐ if you like it!
```

