import requests, time
headers = {'User-Agent': 'Mozilla/5.0', 'Accept': 'application/json', 'Referer': 'https://www.sofascore.com'}

teams = set()
for rnd in range(1, 35):
    url = f'https://api.sofascore.com/api/v1/unique-tournament/238/season/52769/events/round/{rnd}'
    data = requests.get(url, headers=headers, timeout=10).json()
    for e in data.get('events', []):
        teams.add(e['homeTeam']['name'])
        teams.add(e['awayTeam']['name'])
    time.sleep(0.3)

print('Equipas no Sofascore 2023/24:')
for t in sorted(teams):
    print(f'  {repr(t)}')
