from django.core.management.base import BaseCommand

class Command(BaseCommand):
    help = "VISION AI diagnostic — checks Ollama and models"

    def handle(self, *args, **options):
        from ai.services.ollama_client import client
        h = client.healthCheck()
        self.stdout.write("\nVISION AI DIAGNOSTIC\n" + "-"*30)
        self.stdout.write(f"{'[OK]' if h['ollama']['connected'] else '[FAIL]'} Ollama reachable ({h['ollama']['baseUrl']})")
        self.stdout.write(f"{'[OK]' if h['textModel']['installed'] else '[FAIL]'} Text model configured: {h['textModel']['name'] or '(none)'} {'installed' if h['textModel']['installed'] else 'NOT installed'}")
        vm = h['visionModel']
        if not vm['configured']:
            self.stdout.write("[INFO] Vision model not configured (set OLLAMA_VISION_MODEL)")
        elif vm['installed'] and vm['capable']:
            self.stdout.write(f"[OK] Vision model configured & installed: {vm['name']} (vision capable)")
        elif vm['installed']:
            self.stdout.write(f"[FAIL] Vision model {vm['name']} installed but not vision-capable")
        else:
            self.stdout.write(f"[FAIL] Vision model configured {vm['name']} NOT installed -> ollama pull {vm['name']}")
        ready = h['ollama']['connected'] and h['textModel']['installed'] and (not vm['configured'] or (vm['installed'] and vm['capable']))
        if ready:
            self.stdout.write("\nVISION is ready for multimodal requests.")
        else:
            self.stdout.write("\nVISION not fully ready — fix above.")
