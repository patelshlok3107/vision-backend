"""
Django settings for VISION backend.
All sensitive and environment-specific values are loaded from .env via python-decouple.
"""

from pathlib import Path
from datetime import timedelta
from urllib.parse import urlparse
from decouple import config, Csv
import os

BASE_DIR = Path(__file__).resolve().parent.parent

# -------------------------------------------------------------------------
# Core settings
# -------------------------------------------------------------------------

SECRET_KEY = config('SECRET_KEY', default='django-insecure-CHANGE-IN-PRODUCTION')
DEBUG = config('DEBUG', default=True, cast=bool)
ALLOWED_HOSTS = config('ALLOWED_HOSTS', default='localhost,127.0.0.1', cast=Csv())

# -------------------------------------------------------------------------
# Installed apps
# -------------------------------------------------------------------------

INSTALLED_APPS = [
    'daphne',
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',

    # Third-party
    'channels',
    'rest_framework',
    'rest_framework_simplejwt',
    'corsheaders',

    # Local apps
    'users',
    'ai_agent',
    'learning',
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'corsheaders.middleware.CorsMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'config.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'config.wsgi.application'
ASGI_APPLICATION = 'config.asgi.application'

# -------------------------------------------------------------------------
# Channels
# -------------------------------------------------------------------------
CHANNEL_LAYERS = {
    'default': {
        'BACKEND': 'channels_redis.core.RedisChannelLayer',
        'CONFIG': {
            'hosts': [config('REDIS_URL', default='redis://localhost:6379')],
        },
    },
}

# -------------------------------------------------------------------------
# Database
# -------------------------------------------------------------------------

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': config('DB_NAME', default='vision_db'),
        'USER': config('DB_USER', default='vision_user'),
        'PASSWORD': config('DB_PASSWORD', default='vision_password'),
        'HOST': config('DB_HOST', default='localhost'),
        'PORT': config('DB_PORT', default='5433'),
    }
}
# Render / Railway provide DATABASE_URL — override if set
if os.environ.get("DATABASE_URL"):
    u = urlparse(os.environ["DATABASE_URL"])
    DATABASES["default"] = {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': u.path[1:],
        'USER': u.username,
        'PASSWORD': u.password,
        'HOST': u.hostname,
        'PORT': u.port or 5432,
    }

AUTH_USER_MODEL = 'users.User'

# -------------------------------------------------------------------------
# Django REST Framework + JWT
# -------------------------------------------------------------------------

REST_FRAMEWORK = {
    'DEFAULT_AUTHENTICATION_CLASSES': (
        'rest_framework_simplejwt.authentication.JWTAuthentication',
    ),
    'DEFAULT_PERMISSION_CLASSES': (
        'rest_framework.permissions.IsAuthenticated',
    ),
    'DEFAULT_RENDERER_CLASSES': (
        'rest_framework.renderers.JSONRenderer',
    ),
}

SIMPLE_JWT = {
    'ACCESS_TOKEN_LIFETIME': timedelta(days=1),
    'REFRESH_TOKEN_LIFETIME': timedelta(days=7),
    'ROTATE_REFRESH_TOKENS': False,
    'BLACKLIST_AFTER_ROTATION': True,
    'AUTH_HEADER_TYPES': ('Bearer',),
}

CORS_ALLOW_ALL_ORIGINS = config('CORS_ALLOW_ALL_ORIGINS', default=True, cast=bool)
CORS_ALLOWED_ORIGINS = config('CORS_ALLOWED_ORIGINS', default='', cast=Csv())
# If explicit origins set, disable allow-all
if CORS_ALLOWED_ORIGINS and CORS_ALLOWED_ORIGINS != ['']:
    CORS_ALLOW_ALL_ORIGINS = False

# -------------------------------------------------------------------------
# Celery / Redis
# -------------------------------------------------------------------------

REDIS_URL = config('REDIS_URL', default='redis://localhost:6379')
CELERY_BROKER_URL = f"{REDIS_URL}/0"
CELERY_RESULT_BACKEND = f"{REDIS_URL}/1"
CELERY_ACCEPT_CONTENT = ['json']
CELERY_TASK_SERIALIZER = 'json'

# Celery beat — simplified (no reminder workers)

# -------------------------------------------------------------------------
# Continuous Learning Default Settings
# -------------------------------------------------------------------------

LEARNING_ENABLED = config('LEARNING_ENABLED', default=True, cast=bool)
LEARNING_SCHEDULE_HOUR = config('LEARNING_SCHEDULE_HOUR', default=2, cast=int)
LEARNING_MIN_QUALITY_SCORE = config('LEARNING_MIN_QUALITY_SCORE', default=60, cast=int)

# -------------------------------------------------------------------------
# Ollama / Local AI configuration
# -------------------------------------------------------------------------

AI_PROVIDER = config('AI_PROVIDER', default='ollama')
LOCAL_AI_ONLY = config('LOCAL_AI_ONLY', default=True, cast=bool)

# Cloud LLM fallback (Groq / OpenAI) — free tier for Render
GROQ_API_KEY = config('GROQ_API_KEY', default='')
OPENAI_API_KEY = config('OPENAI_API_KEY', default='')
GROQ_MODEL = config('GROQ_MODEL', default='llama3-8b-8192')
OPENAI_MODEL = config('OPENAI_MODEL', default='gpt-4o-mini')

OLLAMA_BASE_URL = config('OLLAMA_BASE_URL', default='http://localhost:11434')
# Text model — primary reasoning model
OLLAMA_TEXT_MODEL = config('OLLAMA_TEXT_MODEL', default=config('OLLAMA_MODEL', default='llama3'))
OLLAMA_MODEL = OLLAMA_TEXT_MODEL  # backward alias for legacy code
OLLAMA_VISION_MODEL = config('OLLAMA_VISION_MODEL', default='')
OLLAMA_VISION_ENABLED = config('OLLAMA_VISION_ENABLED', default=True, cast=bool)
# Model router — Phase 1 (all local, fall back to text model)
OLLAMA_FAST_MODEL = config('OLLAMA_FAST_MODEL', default=OLLAMA_TEXT_MODEL)
OLLAMA_THINK_MODEL = config('OLLAMA_THINK_MODEL', default=OLLAMA_TEXT_MODEL)
OLLAMA_CODE_MODEL = config('OLLAMA_CODE_MODEL', default=OLLAMA_TEXT_MODEL)
OLLAMA_EMBEDDING_MODEL = config('OLLAMA_EMBEDDING_MODEL', default='nomic-embed-text')
OLLAMA_EMBEDDING_DIM = config('OLLAMA_EMBEDDING_DIM', default=768, cast=int)
OLLAMA_TIMEOUT = config('OLLAMA_TIMEOUT', default=120, cast=int)
OLLAMA_CONNECT_TIMEOUT = config('OLLAMA_CONNECT_TIMEOUT', default=10, cast=int)
OLLAMA_TEXT_TIMEOUT = config('OLLAMA_TEXT_TIMEOUT', default=90000, cast=int)
# 5 minutes for slow vision models downloading/loading to GPU
OLLAMA_VISION_TIMEOUT = config('OLLAMA_VISION_TIMEOUT', default=300000, cast=int)
OLLAMA_KEEP_ALIVE = config('OLLAMA_KEEP_ALIVE', default='30m')
OLLAMA_NUM_PREDICT = config('OLLAMA_NUM_PREDICT', default=2048, cast=int)
OLLAMA_NUM_CTX = config('OLLAMA_NUM_CTX', default=8192, cast=int)
OLLAMA_TEMPERATURE = config('OLLAMA_TEMPERATURE', default=0.2, cast=float)

# -------------------------------------------------------------------------
# ULTRA FAST MODE — latency-first defaults (can override via .env)
# -------------------------------------------------------------------------
ULTRA_FAST_NUM_CTX = config('ULTRA_FAST_NUM_CTX', default=1024, cast=int)
ULTRA_FAST_NUM_PREDICT = config('ULTRA_FAST_NUM_PREDICT', default=256, cast=int)
ULTRA_FAST_TEMPERATURE = config('ULTRA_FAST_TEMPERATURE', default=0.4, cast=float)
ULTRA_FAST_HISTORY_MESSAGES = config('ULTRA_FAST_HISTORY_MESSAGES', default=3, cast=int)
ULTRA_FAST_TOP_K = config('ULTRA_FAST_TOP_K', default=20, cast=int)
ULTRA_FAST_TOP_P = config('ULTRA_FAST_TOP_P', default=0.9, cast=float)
ULTRA_FAST_REPEAT_PENALTY = config('ULTRA_FAST_REPEAT_PENALTY', default=1.1, cast=float)
NORMAL_NUM_CTX = config('NORMAL_NUM_CTX', default=4096, cast=int)
NORMAL_NUM_PREDICT = config('NORMAL_NUM_PREDICT', default=1024, cast=int)
CODE_NUM_CTX = config('CODE_NUM_CTX', default=16384, cast=int)
CODE_NUM_PREDICT = config('CODE_NUM_PREDICT', default=4096, cast=int)
THINK_NUM_CTX = config('THINK_NUM_CTX', default=12288, cast=int)
THINK_NUM_PREDICT = config('THINK_NUM_PREDICT', default=3072, cast=int)
AGENT_NUM_CTX = config('AGENT_NUM_CTX', default=12288, cast=int)
AGENT_NUM_PREDICT = config('AGENT_NUM_PREDICT', default=2048, cast=int)

# VISION attachments
VISION_MAX_IMAGES_PER_MESSAGE = config('VISION_MAX_IMAGES_PER_MESSAGE', default=5, cast=int)
VISION_MAX_IMAGE_SIZE_MB = config('VISION_MAX_IMAGE_SIZE_MB', default=10, cast=int)
VISION_MAX_IMAGE_DIMENSION = config('VISION_MAX_IMAGE_DIMENSION', default=2048, cast=int)

# Phase 2: workspace sandbox (per-user isolated)
VISION_WORKSPACE_ROOT = config('VISION_WORKSPACE_ROOT', default=str(BASE_DIR / "workspace"))
# Tools: allow web search when local only? Still allowed via mock fallback
VISION_TOOLS_ENABLED = config('VISION_TOOLS_ENABLED', default=True, cast=bool)
VISION_TERMINAL_ENABLED = config('VISION_TERMINAL_ENABLED', default=True, cast=bool)

# -------------------------------------------------------------------------
# File uploads
# -------------------------------------------------------------------------

MEDIA_URL = '/media/'
MEDIA_ROOT = BASE_DIR / 'media'
DATA_UPLOAD_MAX_MEMORY_SIZE = 26214400  # 25 MB

# -------------------------------------------------------------------------
# Password validation
# -------------------------------------------------------------------------

AUTH_PASSWORD_VALIDATORS = [
    {'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator'},
    {'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator'},
    {'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator'},
    {'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator'},
]

# -------------------------------------------------------------------------
# Internationalization
# -------------------------------------------------------------------------

LANGUAGE_CODE = 'en-us'
TIME_ZONE = 'UTC'
USE_I18N = True
USE_TZ = True

STATIC_URL = 'static/'
STATIC_ROOT = BASE_DIR / "staticfiles"
DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'
