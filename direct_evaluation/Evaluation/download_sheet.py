import os
import argparse
import io
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from googleapiclient.errors import HttpError
from typing import Optional, Tuple

SCOPES = ['https://www.googleapis.com/auth/drive']
MIMETYPE_EXCEL = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
MIMETYPE_SHEETS = 'application/vnd.google-apps.spreadsheet'

def get_drive_service(credentials_file: str):
    """Authenticates and returns the Google Drive service object."""
    try:
        creds = service_account.Credentials.from_service_account_file(
            credentials_file, scopes=SCOPES)
        return build('drive', 'v3', credentials=creds)
    except Exception as e:
        print(f"[ERROR] Authentication failed. Check your credentials file ({credentials_file}). Error: {e}")
        return None

def find_sheet(service, sheet_name: str, parent_folder_id: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    """Searches for an existing Google Sheet by name."""
    query = f"name='{sheet_name}' and mimeType='{MIMETYPE_SHEETS}' and trashed=false"
    
    if parent_folder_id:
        query += f" and '{parent_folder_id}' in parents"

    try:
        response = service.files().list(
            q=query,
            spaces='drive',
            fields='files(id, name)'
        ).execute()
        
        file_list = response.get('files', [])
        if file_list:
            return file_list[0]['id'], file_list[0]['name']
        return None, None
    except HttpError as error:
        print(f'[ERROR] An error occurred during file search: {error}')
        return None, None


def download_sheets_to_excel(service, file_id: str, local_filepath: str) -> bool:
    """Downloads a Google Sheet and converts it to a local Excel file."""
    print(f"🔄 Fetching existing Google Sheet (ID: {file_id}) to {local_filepath}...")
    
    try:
        request = service.files().export(fileId=file_id, mimeType=MIMETYPE_EXCEL)
        fh = io.FileIO(local_filepath, 'wb')
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while done is False:
            status, done = downloader.next_chunk()
        
        print(f"✅ Download complete. Local Excel file updated with sheet data.")
        return True
    except HttpError as error:
        print(f'[ERROR] An error occurred during download of Google Sheet (ID: {file_id}): {error}')
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Downloads a Google Sheet, converting it to a local Excel file for updating."
    )
    parser.add_argument(
        "--excel_path", 
        type=str, 
        default="./GRPO/Evaluation/metrics_summary.xlsx", 
        help="Path to the local Excel file."
    )
    parser.add_argument(
        "--run_name", 
        type=str, 
        required=True, 
        help="Run name (used as fallback for sheet title)."
    )
    parser.add_argument(
        "--credentials", 
        type=str, 
        default="service_account_credentials.json", 
        help="Path to the Google service account JSON key file."
    )
    parser.add_argument(
        "--folder_id", 
        type=str, 
        default=None, 
        help="Optional: ID of the Google Drive folder to search in."
    )
    parser.add_argument(
        "--google_sheet_name", 
        type=str, 
        default=None, 
        help="The fixed name of the Google Sheet file (overrides derived name)."
    )
    args = parser.parse_args()

    # Use the explicit sheet name if provided, otherwise derive it from run_name
    if args.google_sheet_name:
        sheet_title = args.google_sheet_name
    else:
        sheet_title = f"Metrics Summary - {args.run_name}"

    if not os.path.exists(args.credentials):
        print(f"Error: Credentials file not found at {args.credentials}. Please follow the Google API Setup instructions.")
        return

    service = get_drive_service(args.credentials)
    if not service:
        return

    file_id, found_name = find_sheet(service, sheet_title, args.folder_id)

    if file_id:
        download_sheets_to_excel(service, file_id, args.excel_path)
    else:
        print(f"Sheet not found: '{sheet_title}'. A new sheet will be created on first upload.")

if __name__ == '__main__':
    main()