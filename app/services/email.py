"""
app/services/email.py – Email sending service using SMTP
"""
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from app.config import get_settings

settings = get_settings()


async def send_reset_password_email(email_to: str, reset_url: str) -> None:
    """Send a password reset email to the user."""

    subject = "Reset your Scanalyze password"

    html_content = f"""
    <html>
      <body>
        <h2>Reset your password</h2>
        <p>You requested a password reset for your Scanalyze account.</p>
        <p>Click the link below to reset your password. This link expires in 24 hours.</p>
        <a href="{reset_url}" 
           style="background-color:#4F46E5;color:white;padding:12px 24px;
                  text-decoration:none;border-radius:6px;display:inline-block;">
          Reset Password
        </a>
        <p>If you did not request this, ignore this email.</p>
      </body>
    </html>
    """

    # Build the email
    message = MIMEMultipart("alternative")
    message["Subject"] = subject
    message["From"] = settings.EMAIL_FROM
    message["To"] = email_to
    message.attach(MIMEText(html_content, "html"))

    # Send via SMTP
    with smtplib.SMTP(settings.SMTP_HOST, settings.SMTP_PORT) as server:
        server.ehlo()
        server.starttls()
        server.login(settings.SMTP_USER, settings.SMTP_PASSWORD)
        server.sendmail(settings.EMAIL_FROM, email_to, message.as_string())
        
async def send_admin_new_user_notification(admin_email: str, user_email: str, user_full_name: str) -> None:
  """Notify admin that a new user is waiting for approval."""
  subject = "New user registration - Approval required"
  html_content = f"""
  <html>
    <body>
      <h2>New user waiting for approval</h2>
      <p>A new user has registered and is waiting for your approval.</p>
      <table>
        <tr><td><strong>Name:</strong></td><td>{user_full_name}</td></tr>
        <tr><td><strong>Email:</strong></td><td>{user_email}</td></tr>
      </table>
      <p>Please log in to the admin panel to approve or reject this account.</p>
    </body>
  </html>
  """
  message = MIMEMultipart("alternative") # this is the structure of the email
  message["Subject"] = subject
  message["From"] = settings.EMAIL_FROM # = The sender address of all emails sent by the application
  message["To"] = admin_email
  message.attach(MIMEText(html_content, "html"))
  
  with smtplib.SMTP(settings.SMTP_HOST, settings.SMTP_PORT) as server:
      server.ehlo()
      server.starttls()
      server.login(settings.SMTP_USER, settings.SMTP_PASSWORD)
      server.sendmail(settings.EMAIL_FROM, admin_email, message.as_string())
      
async def send_account_approved_email(user_email: str, user_full_name: str) -> None:
  """Notify user that their account has been approved."""
  subject = "Your Scanalyze account has been approved!"
  
  html_content = f"""
  <html>
    <body>
      <h2>Welcome to Scanalyze, {user_full_name}!</h2>
      <p>Your account has been approved by the administrator.</p>
      <p>You can now log in and start using Scanalyze.</p>
      <a href="{settings.FRONTEND_URL}/login"
         style="background-color:#4F46E5;color:white;padding:12px 24px;
                text-decoration:none;border-radius:6px;display:inline-block;">
        Log in now
      </a>
    </body>
  </html>
  """
  message = MIMEMultipart("alternative")
  message["Subject"] = subject
  message["From"] = settings.EMAIL_FROM
  message["To"] = user_email
  message.attach(MIMEText(html_content, "html"))
  
  with smtplib.SMTP(settings.SMTP_HOST, settings.SMTP_PORT) as server:
    server.ehlo()
    server.starttls()
    server.login(settings.SMTP_USER, settings.SMTP_PASSWORD)
    server.sendmail(settings.EMAIL_FROM, user_email, message.as_string())

async def send_account_rejected_email(user_email: str, user_full_name: str) -> None:
  """Notify user that their account has been rejected."""
  subject = "Your Scanalyze account registration has been rejected"
  
  html_content = f"""
  <html>
    <body>
      <h2>Account Registration Update</h2>
      <p>Dear {user_full_name},</p>
      <p>Unfortunately, your account registration has been rejected by the administrator.</p>
      <p>If you think this is a mistake, please contact us.</p>
    </body>
  </html>
  """ 
  message = MIMEMultipart("alternative")
  message["Subject"] = subject
  message["From"] = settings.EMAIL_FROM
  message["To"] = user_email
  message.attach(MIMEText(html_content, "html"))
  
  with smtplib.SMTP(settings.SMTP_HOST, settings.SMTP_PORT) as server:
    server.ehlo()
    server.starttls()
    server.login(settings.SMTP_USER, settings.SMTP_PASSWORD)
    server.sendmail(settings.EMAIL_FROM, user_email, message.as_string())

async def send_email_verification(email_to: str, verify_url: str) -> None:
  """Send an email verification link to the admin"""
  
  subject = "Verify your Scanalyze email address"
  html_content = f"""
  <html>
    <body>
      <h2>Verify your email address</h2>
      <p>Thank you for registering on Scanalyze!</p>
      <p>Please click the button below to verify your email address.</p>
      <p>This link expires in 24 hours.</p>
      <a href="{verify_url}"
        style="background-color:#4F46E5;color:white;padding:12px 24px;
              text-decoration:none;border-radius:6px;display:inline-block;">
        Verify Email
      </a>
      <p>If you did not create an account, ignore this email.</p>
    </body>
  </html>
  """
  message = MIMEMultipart("alternative")
  message["Subject"] = subject
  message["From"] = settings.EMAIL_FROM
  message["To"] = email_to
  message.attach(MIMEText(html_content, "html"))
   
  with smtplib.SMTP(settings.SMTP_HOST, settings.SMTP_PORT) as server:
    server.ehlo()
    server.starttls()
    server.login(settings.SMTP_USER, settings.SMTP_PASSWORD)
    server.sendmail(settings.EMAIL_FROM, email_to, message.as_string())