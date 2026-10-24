from django.contrib.auth import get_user_model
from django.test import TestCase
from django.urls import reverse

User = get_user_model()

GOOD_PASSWORD = "Str0ng-Pass-993"


def register_data(**overrides):
    data = {
        "username": "dhiru",
        "email": "dhiru@example.com",
        "password1": GOOD_PASSWORD,
        "password2": GOOD_PASSWORD,
    }
    data.update(overrides)
    return data


class RegistrationTests(TestCase):
    def test_register_page_loads(self):
        response = self.client.get(reverse("accounts:register"))
        self.assertEqual(response.status_code, 200)

    def test_successful_registration_creates_user_and_logs_in(self):
        response = self.client.post(reverse("accounts:register"), register_data())
        self.assertRedirects(response, reverse("predictor:dashboard"))
        self.assertTrue(User.objects.filter(username="dhiru").exists())
        # The new user is logged in straight away
        self.assertIn("_auth_user_id", self.client.session)

    def test_password_is_stored_hashed(self):
        self.client.post(reverse("accounts:register"), register_data())
        user = User.objects.get(username="dhiru")
        self.assertNotEqual(user.password, GOOD_PASSWORD)
        self.assertTrue(user.check_password(GOOD_PASSWORD))

    def test_duplicate_email_is_rejected(self):
        User.objects.create_user("first", "dhiru@example.com", GOOD_PASSWORD)
        response = self.client.post(
            reverse("accounts:register"), register_data(username="second")
        )
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "already exists")
        self.assertFalse(User.objects.filter(username="second").exists())

    def test_mismatched_passwords_are_rejected(self):
        response = self.client.post(
            reverse("accounts:register"), register_data(password2="Different-Pass-1")
        )
        self.assertEqual(response.status_code, 200)
        self.assertFalse(User.objects.filter(username="dhiru").exists())

    def test_weak_password_is_rejected(self):
        response = self.client.post(
            reverse("accounts:register"),
            register_data(password1="12345678", password2="12345678"),
        )
        self.assertEqual(response.status_code, 200)
        self.assertFalse(User.objects.filter(username="dhiru").exists())

    def test_logged_in_user_is_sent_away_from_register(self):
        User.objects.create_user("dhiru", "dhiru@example.com", GOOD_PASSWORD)
        self.client.login(username="dhiru", password=GOOD_PASSWORD)
        response = self.client.get(reverse("accounts:register"))
        self.assertRedirects(response, reverse("predictor:dashboard"))


class LoginLogoutTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user("dhiru", "dhiru@example.com", GOOD_PASSWORD)

    def test_login_page_loads(self):
        response = self.client.get(reverse("accounts:login"))
        self.assertEqual(response.status_code, 200)

    def test_login_with_correct_password(self):
        response = self.client.post(
            reverse("accounts:login"),
            {"username": "dhiru", "password": GOOD_PASSWORD},
        )
        self.assertRedirects(response, reverse("predictor:dashboard"))
        self.assertIn("_auth_user_id", self.client.session)

    def test_login_with_wrong_password_shows_error(self):
        response = self.client.post(
            reverse("accounts:login"),
            {"username": "dhiru", "password": "wrong-password"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "correct username and password")
        self.assertNotIn("_auth_user_id", self.client.session)

    def test_logout_requires_post_and_ends_session(self):
        self.client.login(username="dhiru", password=GOOD_PASSWORD)
        # A plain GET must not log the user out
        self.assertEqual(self.client.get(reverse("accounts:logout")).status_code, 405)
        response = self.client.post(reverse("accounts:logout"))
        self.assertRedirects(response, reverse("predictor:home"))
        self.assertNotIn("_auth_user_id", self.client.session)

    def test_dashboard_requires_login(self):
        url = reverse("predictor:dashboard")
        response = self.client.get(url)
        self.assertRedirects(response, f"{reverse('accounts:login')}?next={url}")

    def test_login_redirects_back_to_requested_page(self):
        url = reverse("predictor:history")
        response = self.client.post(
            reverse("accounts:login"),
            {"username": "dhiru", "password": GOOD_PASSWORD, "next": url},
        )
        self.assertRedirects(response, url)
