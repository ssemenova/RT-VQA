import sqlite3

class Cache(object):

  def __init__(self, db_name, cache_size, evict_mod, use_ram):
    self.use_ram = use_ram
    self.cache_size = cache_size
    self.evict_mod = evict_mod

    self.oldest_id = 0

    if use_ram:
      self.db = {}
    else:
      self.db_name = db_name + '.db'
      self.conn = _connect()
      cursor = self.conn.cursor()    
      cursor.execute("create table videochunks(id, vgg, c3d)")
  
  def _connect():
    return sqlite3.connect(db_name + '.db')

  def get(id):
    if self.use_ram:
      chunk = self.db.get(id)
      return chunk.vgg, chunk.c3d
    else:
      cursor = self.conn.cursor()
      t = (id,)
      cursor.execute('SELECT * FROM videochunks WHERE id=?', t)
      results = cursor.fetchone()
      return results[1], results[2]

  def insert(id, vgg, c3d):
    # TODO: is self a valid way to insert vgg and c3d data?

    if self.use_ram:
      self.db.update({
        id=Chunk()
      })
    else:
      cursor = self.conn.cursor()
      cursor.execute(
        "INSERT INTO videochunks VALUES(" +
        "'" + id + "', '" + vgg + "', '" + c3d + "'"
        + ")"
      )
      self.conn.commit()

    # Loosely evict from the cache every [evict_mod] chunks
    oldest_allowed_id = id - cache_size
    should_evict = oldest_allowed_id % evict_mod == 0
    if should_evict:
      _evict(oldest_allowed_id)
      self.oldest_id = oldest_allowed_id

  def _evict(oldest_allowed_id):
    if self.use_ram:
      keep_evicting = True
      current_id = oldest_allowed_id + 1

      while keep_evicting:
        if not self.db.pop(current_id[, None]):
          keep_evicting = False
        else:
          current_id += 1
    else:
      cursor = self.conn.cursor()
      t = (oldest_allowed_id,)
      cursor.execute('DELETE FROM videochunks WHERE id<', t)
      self.conn.commit()

      return results[1], results[2]

  
