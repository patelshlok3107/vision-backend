from django.contrib.auth import get_user_model
from django.utils import timezone
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import AllowAny, IsAuthenticated
from django.shortcuts import get_object_or_404
from .models import Profile, UserSettings

User = get_user_model()


class RegisterView(APIView):
    permission_classes = [AllowAny]

    def post(self, request):
        name = request.data.get('name', '').strip()
        email = request.data.get('email', '').strip()
        password = request.data.get('password', '')

        if not email or not password:
            return Response({'detail': 'Email and password are required.'}, status=400)

        if User.objects.filter(email=email).exists():
            return Response({'email': ['A user with this email already exists.']}, status=400)

        first, *last = name.split(' ', 1) if name else ('', '')
        # Generate unique username from email prefix
        base_username = email.split('@')[0]
        username = base_username
        counter = 1
        while User.objects.filter(username=username).exists():
            username = f"{base_username}{counter}"
            counter += 1

        user = User.objects.create_user(
            username=username,
            email=email,
            password=password,
            first_name=first,
            last_name=last[0] if last else '',
        )
        return Response({
            'id': str(user.id),
            'email': user.email,
            'name': user.get_full_name(),
        }, status=201)


class MeView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        u = request.user
        profile, _ = Profile.objects.get_or_create(user=u)
        avatar_url = request.build_absolute_uri(profile.avatar.url) if profile.avatar else None
        return Response({
            'id': str(u.id),
            'email': u.email,
            'username': u.username,
            'name': u.get_full_name() or u.username,
            'first_name': u.first_name,
            'last_name': u.last_name,
            'bio': profile.bio or "",
            'avatar': avatar_url,
            'date_joined': u.date_joined.isoformat() if u.date_joined else None,
        })


class ProfileView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        u = request.user
        profile, _ = Profile.objects.get_or_create(user=u)
        avatar_url = request.build_absolute_uri(profile.avatar.url) if profile.avatar else None
        return Response({
            'id': str(u.id),
            'email': u.email,
            'username': u.username,
            'name': u.get_full_name() or u.username,
            'first_name': u.first_name,
            'last_name': u.last_name,
            'bio': profile.bio or "",
            'avatar': avatar_url,
            'date_joined': u.date_joined.isoformat() if u.date_joined else None,
        })

    def patch(self, request):
        u = request.user
        profile, _ = Profile.objects.get_or_create(user=u)
        name = request.data.get('name')
        username = request.data.get('username')
        email = request.data.get('email')
        bio = request.data.get('bio')

        if name is not None:
            name = str(name).strip()
            if len(name) < 2:
                return Response({'detail': 'Name must be at least 2 characters'}, status=400)
            if len(name) > 80:
                return Response({'detail': 'Name too long'}, status=400)
            parts = name.split(' ', 1)
            u.first_name = parts[0]
            u.last_name = parts[1] if len(parts) > 1 else ''

        if username is not None:
            username = str(username).strip().lower()
            if not username:
                return Response({'detail': 'Username required'}, status=400)
            if len(username) < 3 or len(username) > 30:
                return Response({'detail': 'Username must be 3-30 characters'}, status=400)
            if not username.replace('_','').replace('.','').isalnum():
                return Response({'detail': 'Username may only contain letters, numbers, . and _'}, status=400)
            if User.objects.exclude(id=u.id).filter(username=username).exists():
                return Response({'detail': 'Username already taken'}, status=400)
            u.username = username

        if email is not None:
            email = str(email).strip().lower()
            if email != u.email:
                if User.objects.exclude(id=u.id).filter(email=email).exists():
                    return Response({'detail': 'Email already in use'}, status=400)
                # basic email validation
                if '@' not in email or '.' not in email:
                    return Response({'detail': 'Invalid email'}, status=400)
                u.email = email

        if bio is not None:
            bio = str(bio).strip()
            if len(bio) > 500:
                return Response({'detail': 'Bio too long (max 500)'}, status=400)
            profile.bio = bio

        u.save()
        profile.save()
        avatar_url = request.build_absolute_uri(profile.avatar.url) if profile.avatar else None
        return Response({
            'id': str(u.id),
            'email': u.email,
            'username': u.username,
            'name': u.get_full_name() or u.username,
            'first_name': u.first_name,
            'last_name': u.last_name,
            'bio': profile.bio or "",
            'avatar': avatar_url,
            'date_joined': u.date_joined.isoformat() if u.date_joined else None,
        })


class AvatarUploadView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        u = request.user
        profile, _ = Profile.objects.get_or_create(user=u)
        f = request.FILES.get('avatar') or request.FILES.get('file')
        if not f:
            return Response({'detail': 'No file uploaded'}, status=400)
        if f.content_type not in ('image/jpeg','image/png','image/webp','image/gif'):
            return Response({'detail': 'Only JPG, PNG, WEBP, GIF allowed'}, status=400)
        if f.size > 5*1024*1024:
            return Response({'detail': 'File too large (max 5MB)'}, status=400)
        # validate image dimensions via PIL
        try:
            from PIL import Image
            img = Image.open(f)
            w,h = img.size
            if w < 64 or h < 64:
                return Response({'detail': 'Image too small (min 64x64)'}, status=400)
            if max(w,h) > 4000:
                return Response({'detail': 'Image too large (max 4000px)'}, status=400)
            f.seek(0)
        except Exception:
            pass
        # remove old
        if profile.avatar:
            try:
                profile.avatar.delete(save=False)
            except: pass
        profile.avatar.save(f.name, f, save=True)
        avatar_url = request.build_absolute_uri(profile.avatar.url)
        return Response({'avatar': avatar_url})

    def delete(self, request):
        u = request.user
        profile, _ = Profile.objects.get_or_create(user=u)
        if profile.avatar:
            try:
                profile.avatar.delete(save=False)
            except: pass
            profile.avatar = None
            profile.save()
        return Response({'deleted': True})


class SettingsView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        s, _ = UserSettings.objects.get_or_create(user=request.user)
        return Response({
            'language': s.language,
            'default_mode': s.default_mode,
            'enter_to_send': s.enter_to_send,
            'show_suggested_prompts': s.show_suggested_prompts,
            'auto_scroll': s.auto_scroll,
            'confirm_delete': s.confirm_delete,
            'theme': s.theme,
            'chat_density': s.chat_density,
            'animations': s.animations,
            'reduce_motion': s.reduce_motion,
            'font_size': s.font_size,
            'voice_enabled': s.voice_enabled,
            'voice_id': s.voice_id,
            'speech_speed': s.speech_speed,
            'autoplay_voice': s.autoplay_voice,
            'chat_model': s.chat_model,
            'code_model': s.code_model,
            'vision_model': s.vision_model,
            'reasoning_model': s.reasoning_model,
            'agent_model': s.agent_model,
            'temperature': s.temperature,
            'context_length': s.context_length,
            'streaming': s.streaming,
            'show_generation_status': s.show_generation_status,
            'fast_mode': s.fast_mode,
            'use_routing': s.use_routing,
            'keep_warm': s.keep_warm,
            'max_tokens': s.max_tokens,
            'chat_history_enabled': s.chat_history_enabled,
            'memory_enabled': s.memory_enabled,
            'save_files': s.save_files,
            'use_history_context': s.use_history_context,
            'analytics': s.analytics,
            'personalization': s.personalization,
            'notif_ai_complete': s.notif_ai_complete,
            'notif_agent_complete': s.notif_agent_complete,
            'notif_build_complete': s.notif_build_complete,
            'notif_research_complete': s.notif_research_complete,
            'notif_system': s.notif_system,
            'notif_email': s.notif_email,
            'updated_at': s.updated_at.isoformat() if s.updated_at else None,
        })

    def patch(self, request):
        s, _ = UserSettings.objects.get_or_create(user=request.user)
        allowed = {f.name for f in UserSettings._meta.get_fields() if hasattr(f,'column')}
        for k,v in request.data.items():
            if k in allowed and k not in ('id','user','created_at','updated_at'):
                # type coercion
                field = UserSettings._meta.get_field(k)
                try:
                    if field.get_internal_type() in ('BooleanField',):
                        v = bool(v) if not isinstance(v,str) else v.lower() not in ('0','false','off','no')
                    elif field.get_internal_type() == 'FloatField':
                        v = float(v)
                    elif field.get_internal_type() == 'IntegerField':
                        v = int(v)
                    setattr(s, k, v)
                except Exception as e:
                    return Response({'detail': f'Invalid value for {k}: {e}'}, status=400)
        s.save()
        return Response({'saved': True})


class ChangePasswordView(APIView):
    permission_classes = [IsAuthenticated]
    def post(self, request):
        cur = request.data.get('current_password') or request.data.get('old_password') or ''
        new = request.data.get('new_password') or request.data.get('password') or ''
        confirm = request.data.get('confirm_password') or request.data.get('confirm') or ''
        if not cur or not new:
            return Response({'detail': 'Current and new password required'}, status=400)
        if not request.user.check_password(cur):
            return Response({'detail': 'Current password is incorrect'}, status=400)
        if len(new) < 8:
            return Response({'detail': 'New password must be at least 8 characters'}, status=400)
        if confirm and new != confirm:
            return Response({'detail': 'Passwords do not match'}, status=400)
        request.user.set_password(new)
        request.user.save()
        return Response({'changed': True})


class DataStatsView(APIView):
    permission_classes = [IsAuthenticated]
    def get(self, request):
        from ai_agent.models import Conversation, Memory, Attachment
        from django.db.models import Sum
        convs = Conversation.objects.filter(user=request.user).count()
        mems = Memory.objects.filter(user=request.user).count()
        files = Attachment.objects.filter(user=request.user).count()
        total_size = Attachment.objects.filter(user=request.user).aggregate(s=Sum('file_size'))['s'] or 0
        # estimate storage: files + messages content
        return Response({
            'conversations': convs,
            'memory_items': mems,
            'files': files,
            'storage_bytes': total_size,
            'storage_mb': round(total_size/1024/1024, 2),
        })


class ExportDataView(APIView):
    permission_classes = [IsAuthenticated]
    def get(self, request):
        from ai_agent.models import Conversation, Memory, Message
        import json as _j
        from django.http import JsonResponse
        convs = Conversation.objects.filter(user=request.user).prefetch_related('messages')
        data = {
            'exported_at': timezone.now().isoformat(),
            'user': {'email': request.user.email, 'username': request.user.username, 'name': request.user.get_full_name()},
            'conversations': [
                {
                    'id': str(c.id),
                    'title': c.title,
                    'created_at': c.created_at.isoformat() if c.created_at else None,
                    'messages': [{'role': m.role, 'content': m.content, 'created_at': m.created_at.isoformat() if m.created_at else None} for m in c.messages.all()]
                } for c in convs
            ],
            'memories': list(Memory.objects.filter(user=request.user).values('category','content','importance','is_pinned','created_at')),
        }
        return Response(data)


class DeleteAccountView(APIView):
    permission_classes = [IsAuthenticated]
    def post(self, request):
        # require password confirmation
        pwd = request.data.get('password') or ''
        confirm = request.data.get('confirm') or ''
        if confirm != 'DELETE':
            return Response({'detail': 'Please type DELETE to confirm'}, status=400)
        if not request.user.check_password(pwd):
            return Response({'detail': 'Password incorrect'}, status=400)
        # delete user (cascades)
        request.user.delete()
        return Response({'deleted': True})


class ClearHistoryView(APIView):
    permission_classes = [IsAuthenticated]
    def post(self, request):
        from ai_agent.models import Conversation
        deleted, _ = Conversation.objects.filter(user=request.user).delete()
        return Response({'deleted': deleted})
