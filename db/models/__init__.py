from .user import User
from .department import Department
from .site_hierarchy import SiteHierarchy
from .site_location import SiteLocation
from .camera import Camera
from .access_group import AccessGroup
from .member import Member

# tables that depend on User
from .activity_log import ActivityLog
from .notification import Notification
from .member_embedding import MemberEmbedding
from .normalized_data import NormalizedData

# association tables
from .associations import site_location_access, member_access

