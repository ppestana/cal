import requests, time
headers = {'User-Agent': 'Mozilla/5.0', 'Accept': 'application/json', 'Referer': 'https://www.sofascore.com'}

teams = set()
for rnd in range(1, 35):
    url = 'https://api.sofascore.com/api/v1/unique-tournament/238/season/13539/events/round/' + str(rnd)
    data = requests.get(url, headers=headers, timeout=10).json()
    for e in data.get('events', []):
        teams.add(e['homeTeam']['name'])
        teams.add(e['awayTeam']['name'])
    time.sleep(0.3)

print('Equipas no Sofascore 2017/18:')
for t in sorted(teams):
    print(' ', repr(t))
